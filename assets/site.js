/* Search over search.json, and the decision-log filter. No dependencies; both degrade to a
   plain page if this file fails to load. */
(function () {
  "use strict";

  var input = document.getElementById("q");
  var results = document.getElementById("results");
  var index = null;
  var selected = -1;

  function base() {
    return input.getAttribute("data-index").replace(/search\.json$/, "");
  }

  function load() {
    if (index !== null) return Promise.resolve(index);
    return fetch(input.getAttribute("data-index"))
      .then(function (r) { return r.json(); })
      .then(function (data) { index = data; return index; })
      .catch(function () { index = []; return index; });
  }

  function score(entry, needle) {
    var title = entry.t.toLowerCase();
    if (title === needle) return 100;
    if (title.indexOf(needle) === 0) return 80;
    if (title.indexOf(needle) > -1) return 60;
    for (var i = 0; i < entry.h.length; i++) {
      if (entry.h[i].t.toLowerCase().indexOf(needle) > -1) return 40 - i * 0.01;
    }
    if ((entry.x || "").toLowerCase().indexOf(needle) > -1) return 20;
    return 0;
  }

  function heading(entry, needle) {
    for (var i = 0; i < entry.h.length; i++) {
      if (entry.h[i].t.toLowerCase().indexOf(needle) > -1) return entry.h[i];
    }
    return null;
  }

  function render(needle) {
    if (!needle) { results.hidden = true; results.innerHTML = ""; return; }
    var hits = [];
    (index || []).forEach(function (entry) {
      var s = score(entry, needle);
      if (s > 0) hits.push({ e: entry, s: s, h: heading(entry, needle) });
    });
    hits.sort(function (a, b) { return b.s - a.s; });
    hits = hits.slice(0, 25);
    if (!hits.length) {
      results.innerHTML = '<div class="empty">Nothing matches “' + escape_(needle) + '”.</div>';
      results.hidden = false;
      return;
    }
    results.innerHTML = hits.map(function (hit, n) {
      var href = base() + hit.e.u + (hit.h ? "#" + hit.h.a : "");
      var sub = hit.h ? hit.h.t : (hit.e.x || "").slice(0, 110);
      return '<a href="' + href + '" data-n="' + n + '">' +
        '<span class="r-sec">' + escape_(hit.e.s || "") + "</span> " +
        "<b>" + escape_(hit.e.t) + "</b>" +
        '<div class="r-sub">' + escape_(sub) + "</div></a>";
    }).join("");
    results.hidden = false;
    selected = -1;
  }

  function escape_(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  if (input) {
    input.addEventListener("input", function () {
      var needle = input.value.trim().toLowerCase();
      if (needle.length < 2) { render(""); return; }
      load().then(function () { render(needle); });
    });
    input.addEventListener("keydown", function (event) {
      var links = results.querySelectorAll("a");
      if (event.key === "Escape") { render(""); input.blur(); return; }
      if (!links.length) return;
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        selected += event.key === "ArrowDown" ? 1 : -1;
        selected = Math.max(0, Math.min(links.length - 1, selected));
        links.forEach(function (a) { a.classList.remove("sel"); });
        links[selected].classList.add("sel");
        links[selected].scrollIntoView({ block: "nearest" });
      } else if (event.key === "Enter" && selected > -1) {
        event.preventDefault();
        window.location.href = links[selected].getAttribute("href");
      }
    });
    document.addEventListener("click", function (event) {
      if (!results.contains(event.target) && event.target !== input) render("");
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "/" && document.activeElement !== input) {
        event.preventDefault();
        input.focus();
      }
    });
  }

  // One filter box over every table on the page that asked to be filtered. Used by the decision
  // log and the test index, both long enough that scrolling is not a way to find anything.
  var filter = document.getElementById("table-filter");
  if (filter) {
    filter.addEventListener("input", function () {
      var needle = filter.value.trim().toLowerCase();
      var rows = document.querySelectorAll("table.filterable tbody tr");
      rows.forEach(function (row) {
        row.hidden = needle !== "" && row.textContent.toLowerCase().indexOf(needle) === -1;
      });
    });
  }

  // ---- the front page's wipe ----------------------------------------------------------------
  // Two pixel-aligned stills, one clipped to a fraction of the width. The range input is the
  // control -- it comes free with pointer, touch and keyboard, and a screen reader announces it --
  // and dragging the pane just writes into it. If this file never loads, the CSS default leaves
  // the wipe at half and the figure still reads as a split comparison.
  var compare = document.querySelector(".compare");
  if (compare) {
    var pane = compare.querySelector(".compare-pane");
    var range = compare.querySelector(".compare-range");

    var set = function (percent) {
      percent = Math.max(0, Math.min(100, percent));
      pane.style.setProperty("--x", percent + "%");
    };

    range.addEventListener("input", function () { set(parseFloat(range.value)); });

    var drag = function (event) {
      var box = pane.getBoundingClientRect();
      if (!box.width) return;
      var percent = ((event.clientX - box.left) / box.width) * 100;
      range.value = String(Math.max(0, Math.min(100, percent)));
      set(percent);
    };

    pane.addEventListener("pointerdown", function (event) {
      pane.setPointerCapture(event.pointerId);
      drag(event);
    });
    pane.addEventListener("pointermove", function (event) {
      if (pane.hasPointerCapture(event.pointerId)) drag(event);
    });
    set(parseFloat(range.value));
  }

  // ---- play on hover ---------------------------------------------------------------------
  // The band strip is poster frames until asked for: four autoplaying clips is 7 MB before the
  // reader has scrolled. Pointer in plays, pointer out rewinds -- but only clips this started,
  // so pressing play on a gallery video and then moving the mouse away does not stop it.
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduced) {
    document.querySelectorAll("[data-hover-play], .shot video").forEach(function (video) {
      video.addEventListener("mouseenter", function () {
        if (!video.paused) return;
        var playing = video.play();
        if (playing && playing.catch) playing.catch(function () {});
        video.dataset.auto = "1";
      });
      video.addEventListener("mouseleave", function () {
        if (video.dataset.auto !== "1") return;
        video.pause();
        video.currentTime = 0;
        delete video.dataset.auto;
      });
      video.addEventListener("click", function () { delete video.dataset.auto; });
    });
  } else {
    // A reader who asked for less motion gets the hero as its own poster frame.
    document.querySelectorAll(".hero-bg").forEach(function (video) {
      video.removeAttribute("autoplay");
      video.pause();
    });
  }

  if (window.renderMathInElement) {
    render_math();
  } else {
    window.addEventListener("load", function () {
      if (window.renderMathInElement) render_math();
    });
  }

  function render_math() {
    var targets = document.querySelectorAll(".prose, .toc, .page-head");
    targets.forEach(function (node) { math_in(node); });
  }

  function math_in(node) {
    window.renderMathInElement(node, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    });
  }
})();
