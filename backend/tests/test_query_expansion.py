import asyncio

from app.config import Settings
from app.services.query_expansion import (
    QueryExpander,
    extract_named_app_search_term,
    is_company_experience_question,
    is_contact_question,
    is_plural_app_role_question,
    is_single_app_role_question,
    resolve_portfolio_intent_search_query,
)
from app.services.question_intent import is_greeting_question, is_general_info_question
from app.services.rag import is_experience_list_question, portfolio_num_predict


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
    assert portfolio_num_predict(settings, "اسرد جميع شهادات جهاد") == 2048
    assert portfolio_num_predict(settings, "ما هي مهارات جهاد؟") == 2048
    assert portfolio_num_predict(settings, "اعطني شوية معلومات عن جهاد") == 2048
    assert portfolio_num_predict(settings, "ما رقم هاتفه؟") == 512


def test_plural_app_role_intent():
    q = "ما هو دور جهاد في التطبيقات المختلفة؟"
    assert is_plural_app_role_question(q)
    assert not is_single_app_role_question(q)
    assert is_experience_list_question(q)
    assert resolve_portfolio_intent_search_query(q) is not None
    assert portfolio_num_predict(Settings(), q) == 2048


def test_single_app_role_intent():
    q = "ما دور جهاد في تطبيق نقيم؟"
    assert is_single_app_role_question(q)
    assert not is_plural_app_role_question(q)
    assert not is_experience_list_question(q)
    assert extract_named_app_search_term(q) == "Nuqayyem"
    assert "Nuqayyem" in resolve_portfolio_intent_search_query(q)
    assert portfolio_num_predict(Settings(), q) == 1024


def test_contact_collaboration_intent():
    q = "اريد التواصل مع جهاد من اجل مشروع"
    assert is_contact_question(q)
    assert not is_experience_list_question(q)
    assert "Contact" in resolve_portfolio_intent_search_query(q)
    assert "WeFix" not in resolve_portfolio_intent_search_query(q)


def test_contact_phone_intent():
    q = "اريد رقم للتواصل"
    assert is_contact_question(q)
    assert "outlook" in resolve_portfolio_intent_search_query(q).lower()


def test_greeting_intent():
    for q in ("مرحبا", "السلام عليكم", "hello", "من انت؟"):
        assert is_greeting_question(q)
    assert not is_greeting_question("اعطني شوية معلومات عن جهاد")
    assert is_general_info_question("اعطني شوية معلومات عن جهاد")
    assert resolve_portfolio_intent_search_query("مرحبا") == (
        "Jehad Abu Awwad Mechatronics Engineer Full Stack Code Fellows experienced"
    )
    assert portfolio_num_predict(Settings(), "مرحبا") == 512
    assert portfolio_num_predict(Settings(), "اعطني شوية معلومات عن جهاد") == 2048


def test_company_experience_intent():
    q = "ما هي خبرة جهاد في فور تيك ؟"
    assert is_company_experience_question(q)
    assert not is_experience_list_question(q)
    assert extract_named_app_search_term(q) == "4Tech"
    intent = resolve_portfolio_intent_search_query(q)
    assert "4Tech" in intent
    assert "WeFix" not in intent
    assert portfolio_num_predict(Settings(), q) == 1024


def test_single_app_role_expand_two_queries():
    settings = Settings(portfolio_query_expansion_enabled=True)
    queries = asyncio.run(
        QueryExpander(settings).expand(
            "ما دور جهاد في تطبيق بلنكس؟",
            portfolio_fast=True,
        )
    )
    assert len(queries) == 2
    assert any("Blinx" in query for query in queries)
