from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.utils.translation import get_language

from .engine.runner import SessionIO, NeedAnswer
from .engine import en, kr

@require_http_methods(["GET"])
def back(request):
    # If nothing to undo, just restart
    if not request.session.get("checkpoints"):
        return redirect("start")

    # Remove the checkpoint for the CURRENT question
    _pop_checkpoint(request.session)

    # Now we want to restore to the previous checkpoint (if any)
    prev = request.session["checkpoints"][-1] if request.session.get("checkpoints") else None

    # Remove last recorded answer (UI history)
    hist = request.session.get("answer_history", [])
    if hist:
        last = hist.pop()
        # Also remove from answers dict
        answers = request.session.get("answers", {})
        answers.pop(last.get("key"), None)
        request.session["answers"] = answers
        request.session["answer_history"] = hist

    # Truncate snippets to previous checkpoint lengths
    if prev:
        request.session["progress_snippets"] = request.session.get("progress_snippets", [])[: prev["snip_len"]]
        request.session["progress_inline"] = request.session.get("progress_inline", [])[: prev["inline_len"]]
    else:
        # No previous checkpoint -> clean slate
        request.session["progress_snippets"] = []
        request.session["progress_inline"] = []

    # Clear final result
    request.session.pop("final_result", None)

    request.session.modified = True
    return redirect("wizard")

def _ensure_checkpoints(session):
    session.setdefault("checkpoints", [])

def _push_checkpoint(session, q_key: str):
    _ensure_checkpoints(session)
    session["checkpoints"].append({
        "key": q_key,
        "snip_len": len(session.get("progress_snippets", [])),
        "inline_len": len(session.get("progress_inline", [])),
    })

def _pop_checkpoint(session):
    _ensure_checkpoints(session)
    if not session["checkpoints"]:
        return None
    return session["checkpoints"].pop()

def _is_intro_card(card: dict) -> bool:
    """
    Returns True if this card is the intro/checklist content we moved to the left column.
    Works for both inline_cards (dicts with title/desc/code) and snippet_cards bundles.
    """
    title = (card.get("title") or "").strip()
    desc = (card.get("desc") or "").strip()

    # Titles we want to remove from Snippets column
    bad_titles = {
        "대화형 통계 검정 선택기",
        "확인 사항",
        "Statistical Test Selector",
        "Notes",
        "Important Notes",
        "Checklist",
    }

    if title in bad_titles:
        return True

    # Extra safety: if the description contains these phrases, treat as intro/checklist
    bad_phrases = [
        "질문에 답해 주세요",
        "이 도구는 자주 사용되는 통계 검정을 선택하는 데 도움",
        "does not replace statistical thinking",
        "answer the questions",
    ]
    return any(p in desc for p in bad_phrases)

def _unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _unique_cards_by_key(cards):
    seen = set()
    out = []
    for c in cards:
        k = c.get("key") or (c.get("title"), c.get("desc"), c.get("code"))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out

def _unique_inline_cards(cards):
    seen = set()
    out = []
    for c in cards:
        key = (c.get("title", ""), c.get("desc", ""), c.get("code", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _engine_for_lang():
    lang = get_language() or "en"
    return kr if lang.startswith("ko") else en


def _snippet_bundle(engine, snippet_key: str):
    if hasattr(engine, "ASSUMPTION_SNIPPETS") and snippet_key in engine.ASSUMPTION_SNIPPETS:
        title, desc, code = engine.ASSUMPTION_SNIPPETS[snippet_key]
        return {"kind": "assumption", "key": snippet_key, "title": title, "desc": desc, "code": code}

    if hasattr(engine, "TEST_SNIPPETS") and snippet_key in engine.TEST_SNIPPETS:
        title, desc, code = engine.TEST_SNIPPETS[snippet_key]
        return {"kind": "test", "key": snippet_key, "title": title, "desc": desc, "code": code}

    return {"kind": "other", "key": snippet_key, "title": snippet_key, "desc": "", "code": ""}


@require_http_methods(["GET"])
def start(request):
    request.session.flush()
    return redirect("wizard")


@require_http_methods(["GET", "POST"])
def wizard(request):
    engine = _engine_for_lang()
    io = SessionIO(request.session)
    engine.bind_session(io)

    if request.method == "POST":
        key = request.POST.get("key", "")
        value = request.POST.get("value", "")
        prompt = request.POST.get("prompt", "")
        label = request.POST.get("label", "")

        if key and value:
            io.set_answer(key, value)

            # Store a user-friendly history for UI (order preserved)
            request.session.setdefault("answer_history", [])
            # Remove existing entry for same key (if user changes answer later)
            request.session["answer_history"] = [x for x in request.session["answer_history"] if x.get("key") != key]
            request.session["answer_history"].append({
                "key": key,
                "prompt": prompt,
                "label": label,
                "value": value,
            })

            request.session.modified = True

        return redirect("wizard")

    try:
        result = engine.run_once()
        request.session["final_result"] = result
        request.session.modified = True
        return redirect("results")
    except NeedAnswer as q:
        _ensure_checkpoints(request.session)
        if not request.session["checkpoints"] or request.session["checkpoints"][-1]["key"] != q.key:
            _push_checkpoint(request.session, q.key)
            request.session.modified = True
        progress_keys = _unique_keep_order(request.session.get("progress_snippets", []))
        snippet_cards = [_snippet_bundle(engine, k) for k in progress_keys]
        snippet_cards = [c for c in snippet_cards if not _is_intro_card(c)]
        inline_cards = request.session.get("progress_inline", [])
        inline_cards = [c for c in inline_cards if not _is_intro_card(c)]
        inline_cards = _unique_inline_cards(inline_cards)
        return render(request, "core/question.html", {
            "q": q,
            "snippet_cards": snippet_cards,
            "inline_cards": inline_cards,
        })


@require_http_methods(["GET"])
def results(request):
    engine = _engine_for_lang()
    result = request.session.get("final_result")
    if not result:
        return redirect("wizard")

    # Progress snippet keys (dedupe keep-order)
    progress_keys = _unique_keep_order(request.session.get("progress_snippets", []))

    # Build progress cards and remove intro items
    all_progress_cards = [_snippet_bundle(engine, k) for k in progress_keys]
    all_progress_cards = [c for c in all_progress_cards if not _is_intro_card(c)]

    # Assumptions only + dedupe
    assumption_progress_cards = [c for c in all_progress_cards if c.get("kind") == "assumption"]
    assumption_progress_cards = _unique_cards_by_key(assumption_progress_cards)

    # Inline cards (dedupe + remove intro)
    inline_cards = request.session.get("progress_inline", [])
    inline_cards = [c for c in inline_cards if not _is_intro_card(c)]
    inline_cards = _unique_inline_cards(inline_cards)

    # Final tests (dedupe keep-order)
    final_tests = _unique_keep_order(result.get("final_tests", []))
    final_test_cards = [_snippet_bundle(engine, k) for k in final_tests]

    return render(request, "core/results.html", {
        "result": result,
        "snippet_cards": assumption_progress_cards,  # assumptions only
        "inline_cards": inline_cards,               # any inline notes you want to show
        "final_test_cards": final_test_cards,
    })