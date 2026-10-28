const palette = ["#4c865b", "#9ebd8a", "#d0ad63", "#81a9a0", "#78956d", "#c5d5ba"];
const numberFormat = new Intl.NumberFormat();
let yearChart;
let houseChart;

function showConnection(ready, text) {
  const status = document.getElementById("connection-status");
  status.textContent = text;
  status.style.color = ready ? "#468153" : "#ad7354";
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function createChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, config);
}

function renderYearChart(trend) {
  if (!trend.length) {
    document.getElementById("year-chart").parentElement.innerHTML =
      '<p class="muted">No usable year values are available in the workbook.</p>';
    return;
  }
  yearChart?.destroy();
  yearChart = createChart("year-chart", {
    type: "line",
    data: {
      labels: trend.map((row) => row.year),
      datasets: [{
        data: trend.map((row) => row.count),
        borderColor: "#4b8559",
        backgroundColor: "rgba(110, 158, 111, .14)",
        pointBackgroundColor: "#4b8559",
        pointBorderColor: "#fff",
        pointBorderWidth: 2,
        pointRadius: 2.5,
        pointHoverRadius: 5,
        borderWidth: 2,
        fill: true,
        tension: .32
      }]
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { displayColors: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#879188", maxTicksLimit: 12, font: { size: 9 } }, border: { display: false } },
        y: { beginAtZero: true, grid: { color: "#eef1ec" }, ticks: { color: "#879188", precision: 0, font: { size: 9 } }, border: { display: false } }
      }
    }
  });
}

function renderHouseChart(houses) {
  const labels = houses.map((row) => row.house);
  const counts = houses.map((row) => row.count);
  houseChart?.destroy();
  houseChart = createChart("house-chart", {
    type: "doughnut",
    data: {
      labels,
      datasets: [{ data: counts, backgroundColor: palette.slice(0, labels.length), borderColor: "#fff", borderWidth: 4, hoverOffset: 4 }]
    },
    options: {
      maintainAspectRatio: false,
      cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${numberFormat.format(ctx.raw)}` } } }
    }
  });
  const legend = document.getElementById("house-legend");
  legend.replaceChildren();
  houses.forEach((row, index) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const dot = document.createElement("i");
    dot.style.backgroundColor = palette[index % palette.length];
    item.append(dot, document.createTextNode(`${row.house} · ${numberFormat.format(row.count)}`));
    legend.append(item);
  });
}

function renderEntities(entities) {
  const root = document.getElementById("entity-content");
  root.replaceChildren();
  const max = Math.max(1, ...entities.map((row) => row.record_count));
  entities.forEach((row) => {
    const item = document.createElement("div");
    item.className = "entity-row";
    const category = document.createElement("span");
    category.className = "entity-category";
    category.textContent = row.category;
    const track = document.createElement("div");
    track.className = "entity-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(row.record_count ? 2 : 0, row.record_count / max * 100)}%`;
    track.append(fill);
    const count = document.createElement("span");
    count.className = "entity-count";
    count.textContent = numberFormat.format(row.record_count);
    const terms = document.createElement("div");
    terms.className = "entity-terms";
    terms.textContent = row.terms.length
      ? row.terms.map((term) => `${term.term} (${numberFormat.format(term.count)})`).join(" · ")
      : "No matching terms in available records";
    item.append(category, track, count, terms);
    root.append(item);
  });
}

function renderWords(words) {
  const root = document.getElementById("word-content");
  root.replaceChildren();
  const max = Math.max(1, ...words.map((row) => row.count));
  words.slice(0, 12).forEach((row) => {
    const item = document.createElement("div");
    item.className = "word-row";
    const label = document.createElement("span");
    label.className = "word-label";
    label.textContent = row.word;
    const bar = document.createElement("div");
    bar.className = "word-bar";
    const fill = document.createElement("span");
    fill.style.width = `${row.count / max * 100}%`;
    bar.append(fill);
    const count = document.createElement("span");
    count.className = "word-count";
    count.textContent = numberFormat.format(row.count);
    item.append(label, bar, count);
    root.append(item);
  });
}

function renderTfidf(terms) {
  const root = document.getElementById("tfidf-content");
  root.replaceChildren();
  if (!terms.length) {
    root.textContent = "Not enough repeated text to calculate TF-IDF terms.";
    root.className = "term-cloud muted";
    return;
  }
  terms.forEach((row) => {
    const chip = document.createElement("span");
    chip.className = "term-chip";
    const term = document.createElement("span");
    term.textContent = row.term;
    const score = document.createElement("small");
    score.textContent = row.score.toFixed(2);
    chip.append(term, score);
    root.append(chip);
  });
}

async function loadTopics() {
  const root = document.getElementById("topic-content");
  try {
    const response = await fetch("/api/topics");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Topics are not available.");
    root.replaceChildren();
    if (!payload.topics.length) {
      root.textContent = "BERTopic did not find any reportable topics in the available text.";
      return;
    }
    payload.topics.slice(0, 12).forEach((row) => {
      const item = document.createElement("div");
      item.className = "topic-row";
      const label = document.createElement("div");
      label.className = "topic-label";
      label.textContent = row.label;
      const terms = document.createElement("span");
      terms.className = "topic-terms";
      terms.textContent = row.keywords.map((word) => word.word).join(" · ");
      label.append(terms);
      const count = document.createElement("span");
      count.className = "topic-count";
      count.textContent = `${numberFormat.format(row.count)} records`;
      item.append(label, count);
      root.append(item);
    });
  } catch (error) {
    root.replaceChildren();
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = error.message;
    root.append(message);
  }
}

function renderAnalytics(payload) {
  const summary = payload.summary;
  setText("stat-records", numberFormat.format(summary.record_count));
  setText("stat-record-foot", `Clean, unique records · ${numberFormat.format(summary.workbook_rows)} workbook rows`);
  setText("stat-houses", numberFormat.format(summary.house_count));
  setText("stat-years", numberFormat.format(summary.year_count));
  setText("stat-year-range", summary.year_start && summary.year_end
    ? `${summary.year_start} – ${summary.year_end}`
    : "No usable years available");
  setText("stat-date-range", summary.date_start && summary.date_end
    ? `${summary.date_start.slice(0, 4)} – ${summary.date_end.slice(0, 4)}`
    : "Not available");
  setText("stat-dated-records", `${numberFormat.format(summary.records_with_date)} records with dates`);
  renderYearChart(payload.year_trend);
  renderHouseChart(payload.houses);
  renderEntities(payload.entities);
  renderWords(payload.top_words);
  renderTfidf(payload.tfidf_terms);
  showConnection(true, "Connected");
}

function renderSearchResults(results, root) {
  root.replaceChildren();
  if (!results.length) return;
  const heading = document.createElement("h3");
  heading.className = "evidence-heading";
  heading.textContent = "Records supporting this response";
  root.append(heading);
  results.forEach((result, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const source = document.createElement("div");
    source.className = "result-source";
    source.textContent = `Source [${index + 1}] · ${result.topic && result.topic !== "Other / Outlier" ? result.topic : "Parliamentary record"}`;
    const top = document.createElement("div");
    top.className = "result-top";
    const details = document.createElement("div");
    const title = document.createElement("h3");
    title.className = "result-title";
    title.textContent = result.title;
    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = [result.house, result.year, result.date].filter(Boolean).join(" · ");
    details.append(title, meta);
    const score = document.createElement("span");
    score.className = "score-pill";
    score.textContent = `${(result.similarity_score * 100).toFixed(1)}% match`;
    top.append(details, score);
    const snippet = document.createElement("p");
    snippet.className = "result-snippet";
    snippet.textContent = result.snippet;
    card.append(source, top, snippet);
    if (result.ministry) {
      const ministry = document.createElement("p");
      ministry.className = "result-meta";
      ministry.textContent = `Ministry: ${result.ministry}`;
      card.append(ministry);
    }
    const tags = document.createElement("div");
    tags.className = "result-tags";
    Object.entries(result.entities).forEach(([category, terms]) => {
      terms.forEach((term) => {
        const tag = document.createElement("span");
        tag.textContent = `${category}: ${term}`;
        tags.append(tag);
      });
    });
    if (tags.childElementCount) card.append(tags);
    root.append(card);
  });
}

function renderAnswer(query, payload) {
  const thread = document.getElementById("conversation-thread");
  const turn = document.createElement("article");
  turn.className = "conversation-turn";

  const user = document.createElement("div");
  user.className = "user-bubble";
  user.textContent = query;

  const answerCard = document.createElement("div");
  answerCard.className = "answer-card";
  const heading = document.createElement("div");
  heading.className = "answer-heading";
  const label = document.createElement("strong");
  const icon = document.createElement("i");
  icon.textContent = "✳";
  label.append(icon, document.createTextNode("Record-grounded response"));
  const scope = document.createElement("span");
  scope.className = "answer-scope";
  scope.textContent = payload.evidence_scope || "";
  heading.append(label, scope);

  const text = document.createElement("p");
  text.className = "answer-text";
  text.textContent = payload.answer;
  answerCard.append(heading, text);

  if (payload.themes?.length || payload.agricultural_terms?.length) {
    const themes = document.createElement("div");
    themes.className = "answer-themes";
    payload.themes?.forEach((item) => {
      const chip = document.createElement("span");
      chip.textContent = `Topic · ${item.label}`;
      themes.append(chip);
    });
    payload.agricultural_terms?.forEach((item) => {
      const chip = document.createElement("span");
      chip.textContent = `Term · ${item.term}`;
      themes.append(chip);
    });
    answerCard.append(themes);
  }
  const evidence = document.createElement("div");
  evidence.className = "search-results";
  renderSearchResults(payload.results, evidence);
  answerCard.append(evidence);
  turn.append(user, answerCard);
  thread.append(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = document.getElementById("search-query").value.trim();
    const message = document.getElementById("search-message");
    const button = document.getElementById("search-button");
    const buttonLabel = button.querySelector(".button-label");
    if (!query) {
      message.textContent = "Ask a question about the parliamentary agriculture records.";
      return;
    }
    message.textContent = "Reading the matching records and their BERTopic themes…";
    button.disabled = true;
    buttonLabel.textContent = "Preparing response";
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=6`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Search failed.");
      renderAnswer(query, payload);
      message.textContent = payload.results.length
        ? `${payload.results.length} source records shown. The response is based on the retrieved evidence.`
        : "No source records were returned.";
    } catch (error) {
      message.textContent = error.message;
    } finally {
      button.disabled = false;
      buttonLabel.textContent = "Ask question";
    }
  });

  document.querySelectorAll(".starter-chip").forEach((button) => {
    button.addEventListener("click", () => {
      const input = document.getElementById("search-query");
      input.value = button.dataset.query || "";
      document.getElementById("search-form").requestSubmit();
    });
  });

  try {
    const response = await fetch("/api/analytics");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to load dataset analytics.");
    renderAnalytics(payload);
  } catch (error) {
    showConnection(false, "Unavailable");
    document.getElementById("connection-status").title = error.message;
    setText("search-message", error.message);
  }
  await loadTopics();
});
