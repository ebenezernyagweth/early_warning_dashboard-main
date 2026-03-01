(function () {
  let io;

  function observe(img) {
    if (!img || img.dataset.observed === "1") return;
    img.dataset.observed = "1";

    if (io) {
      io.observe(img);
    } else {
      // Fallback if IO not available: load immediately
      const realSrc = img.getAttribute('data-src');
      if (realSrc) {
        img.src = realSrc;
        img.removeAttribute('data-src');
      }
      img.classList.remove('lazy');
    }
  }

  function onIntersection(entries, observer) {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const img = entry.target;
      const realSrc = img.getAttribute('data-src');
      if (realSrc) {
        img.src = realSrc;
        img.removeAttribute('data-src');
      }
      img.classList.remove('lazy');
      observer.unobserve(img);
    });
  }

  function initObserver() {
    if ('IntersectionObserver' in window) {
      io = new IntersectionObserver(onIntersection, { rootMargin: '200px' });
    } else {
      io = null; // triggers fallback in observe()
    }
  }

  function initAll() {
    document.querySelectorAll('img.lazy').forEach(observe);
  }

  // Run once when script loads
  initObserver();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

  // Re-run when Dash updates the DOM (images created by callbacks)
  if (window && window.dash_renderer && window.dash_renderer.on) {
    window.dash_renderer.on('afterupdate', initAll);
  }

  // Also watch for any future DOM insertions (belt & suspenders)
  const mo = new MutationObserver(() => initAll());
  mo.observe(document.documentElement || document.body, {
    childList: true,
    subtree: true
  });
})();
