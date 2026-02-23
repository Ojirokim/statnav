document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".copybtn");
  if (!btn) return;

  const id = btn.getAttribute("data-copy-target");
  const el = document.getElementById(id);
  if (!el) return;

  const text = el.innerText;

  try {
    await navigator.clipboard.writeText(text);

    const old = btn.innerText;
    btn.innerText = "Copied!";
    btn.disabled = true;

    setTimeout(() => {
      btn.innerText = old;
      btn.disabled = false;
    }, 1500);
  } catch (err) {
    // fallback: highlight text so user can Ctrl+C
    const range = document.createRange();
    range.selectNodeContents(el);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }
});