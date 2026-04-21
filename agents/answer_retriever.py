"""
Answer Retriever: Retrieves form answers via semantic module search.
Path: question → fb_modules (semantic) → fb_forms (published)
      → fb_translation_json → question_ids → fb_{module_id} answers
"""

import json
from loguru import logger
from core.database import db_manager
from core.state import AgentState
from tools.vector_store import module_retriever


class AnswerRetriever:

    def retrieve_answers(self, state: AgentState) -> dict:
        question = state["question"]
        logger.info(f"ANSWER RETRIEVER: Starting path for: {question}")

        # ── Step 1: Semantic search → get module_id ──────────────
        try:
            results = module_retriever.search(question, k=1)
            if not results:
                return {"error": "No matching module found", "should_retry": False}

            module_id = results[0]["module_id"]
            module_name = results[0]["name"]
            similarity_score = results[0]["score"]
            logger.info(f"Matched module: {module_name} (id: {module_id}, score: {similarity_score:.4f})")

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"error": str(e), "should_retry": False}

        # ── Step 2: Get published form_id via module_id ──────────
        form_sql = f"""
            SELECT f.id AS form_id, f.translations_id
            FROM fb_forms f
            WHERE f.module_id = '{module_id}'
              AND f.status = 'PUBLISHED'
            LIMIT 1;
        """
        try:
            result, error, _ = db_manager.execute_query(form_sql, timeout=10)
            if error or not result:
                return {
                    "error": f"No published form found for module: {module_name}",
                    "should_retry": False
                }

            row = dict(result[0]._mapping)
            form_id = str(row["form_id"])
            translations_id = str(row["translations_id"])
            logger.info(f"Found published form: {form_id}")

        except Exception as e:
            logger.error(f"Form lookup failed: {e}")
            return {"error": str(e), "should_retry": False}

        # ── Step 3: Get question_ids from translation JSONB ───────
        questions_sql = f"""
            SELECT elem->>'elementId' AS question_id,
                   elem->>'translatedText' AS question_label
            FROM fb_translation_json tj,
                 jsonb_array_elements(tj.translations) AS elem
            WHERE tj.id = '{translations_id}'
              AND elem->>'language' = 'eng'
              AND elem->>'attribute' = 'NAME'
              AND elem->>'entityType' = 'QUESTION';
        """
        try:
            result, error, _ = db_manager.execute_query(questions_sql, timeout=10)
            if error or not result:
                return {
                    "error": "No questions found in translation JSON",
                    "should_retry": False
                }

            question_ids = set()
            question_labels = {}
            for r in result:
                rd = dict(r._mapping)
                qid = rd["question_id"]
                question_ids.add(qid)
                question_labels[qid] = rd["question_label"]

            logger.info(f"Found {len(question_ids)} questions")

        except Exception as e:
            logger.error(f"Question ID retrieval failed: {e}")
            return {"error": str(e), "should_retry": False}

        # ── Step 4: Query answers from fb_{module_id} ─────────────
        table_name = f"fb_{module_id.replace('-', '_')}"

        answers_sql = f"""
            SELECT answer_data
            FROM {table_name}
            LIMIT 100;
        """
        try:
            result, error, _ = db_manager.execute_query(answers_sql, timeout=30)
            if error:
                logger.error(f"Answer table query failed: {error}")
                return {"error": error, "should_retry": False}

            answers = []
            for row in result:
                row_dict = dict(row._mapping)
                answer_data = row_dict.get("answer_data", {})

                # Parse if string
                if isinstance(answer_data, str):
                    try:
                        answer_data = json.loads(answer_data)
                    except Exception:
                        continue

                # Extract and match answers to question_ids
                raw_answers = answer_data.get("answers", [])
                matched = [
                    {
                        "questionId": a.get("questionId"),
                        "question_label": question_labels.get(
                            a.get("questionId"), "Unknown"
                        ),
                        "answer": a.get("answer", a.get("answerId", ""))
                    }
                    for a in raw_answers
                    if a.get("questionId") in question_ids
                ]

                if matched:
                    answers.append(matched)

            logger.info(f"Retrieved {len(answers)} answer rows")

            # Build preview
            preview = f"Module: {module_name}\n"
            preview += f"Form ID: {form_id}\n"
            preview += f"Questions: {len(question_ids)}\n"
            preview += f"Answer rows: {len(answers)}\n\n"
            if answers:
                preview += "Sample:\n"
                for item in answers[0][:3]:
                    preview += f"  {item['question_label']}: {item['answer']}\n"

            return {
                "query_result": answers,
                "result_preview": preview,
                "sql_query": answers_sql,
                "relevant_tables": [table_name],
                "error": None,
                "should_retry": False
            }

        except Exception as e:
            logger.error(f"Answer retrieval failed: {e}")
            return {"error": str(e), "should_retry": False}


def answer_retriever_node(state: AgentState) -> dict:
    """LangGraph node wrapper for AnswerRetriever."""
    agent = AnswerRetriever()
    return agent.retrieve_answers(state)