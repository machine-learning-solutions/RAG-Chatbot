import asyncio

from app.config import Settings
from app.services.query_expansion import QueryExpander, resolve_portfolio_intent_search_query
from app.services.rag import portfolio_num_predict


def test_resolve_cert_intent():
    q = resolve_portfolio_intent_search_query("ما هي شهادات جهاد؟")
    assert q is not None
    assert "Certification" in q


def test_resolve_skills_intent():
    q = resolve_portfolio_intent_search_query("ما هي مهارات جهاد؟")
    assert q is not None
    assert "Skills" in q


def test_portfolio_expand_caps_to_two_queries():
    settings = Settings(portfolio_query_expansion_enabled=True)
    expander = QueryExpander(settings)
    queries = asyncio.run(
        expander.expand("ما هي مهارات جهاد؟", portfolio_fast=True)
    )
    assert len(queries) == 2
    assert any("Skills" in query for query in queries)


def test_portfolio_num_predict_tiers():
    settings = Settings()
    assert portfolio_num_predict(settings, "اسرد جميع شهادات جهاد") == 1536
    assert portfolio_num_predict(settings, "ما هي مهارات جهاد؟") == 1024
    assert portfolio_num_predict(settings, "ما رقم هاتفه؟") == 512
