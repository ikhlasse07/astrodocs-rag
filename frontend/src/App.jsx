import { useEffect, useRef, useState } from "react";
import SpaceBackground from "./components/SpaceBackground.jsx";
import { askQuestion, getHealth } from "./services/api.js";
import "./App.css";

const EXAMPLES = [
  "How does the Sun produce energy?",
  "What is the purpose of MOXIE?",
  "What is the main goal of Europa Clipper?",
];

const DOCUMENTS = [
  {
    title: "Solar System Lithographs",
    file: "Solar_System_Lithograph_Set.pdf",
  },
  {
    title: "Mars 2020",
    file: "44738_Mars2020_Fact_Sheet_REV_BI.pdf",
  },
  {
    title: "Europa Clipper",
    file: "Europa-Clipper-Fact-Sheet.pdf",
  },
  {
    title: "New Horizons",
    file: "nh-fact-sheet-2015_1.pdf",
  },
];

const ANSWER_STATES = {
  verified_excerpt: {
    label: "Source-matched excerpt",
    tone: "matched",
  },
  abstained: {
    label: "Not enough information",
    tone: "abstained",
  },
  needs_review: {
    label: "Review needed",
    tone: "review",
  },
};

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}s` : "—";
}

function AnswerText({ answer, sources }) {
  const sourceIds = new Set(
    sources.map((source) => source.source_id),
  );

  const parts = answer.split(/(\[\d+\])/g);

  return (
    <p className="answer-copy">
      {parts.map((part, index) => {
        const match = part.match(/^\[(\d+)\]$/);
        const sourceId = match ? Number(match[1]) : null;

        if (sourceId !== null && sourceIds.has(sourceId)) {
          return (
            <a
              className="citation"
              href={`#source-${sourceId}`}
              key={`citation-${index}`}
              aria-label={`Read source ${sourceId}`}
              onClick={() => {
                const card = document.getElementById(
                  `source-${sourceId}`,
                );

                if (card) {
                  card.open = true;
                }
              }}
            >
              {part}
            </a>
          );
        }

        return <span key={`text-${index}`}>{part}</span>;
      })}
    </p>
  );
}

function SourceCard({ source }) {
  return (
    <details
      className="source-card"
      id={`source-${source.source_id}`}
    >
      <summary>
        <span className="source-heading">
          <span className="source-number">
            [{source.source_id}]
          </span>

          <span className="source-file">
            {source.source}
          </span>
        </span>

        <span className="source-meta">
          PDF page {source.page} · Chunk {source.chunk_index}
        </span>

        <span className="passage-label">
          View source passage
        </span>
      </summary>

      <p className="source-text">{source.text}</p>
    </details>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [health, setHealth] = useState(null);
  const [connection, setConnection] = useState("checking");

  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const requestController = useRef(null);
  const requestRunning = useRef(false);
  const requestStarted = useRef(0);

  useEffect(() => {
    const controller = new AbortController();

    getHealth(controller.signal)
      .then((data) => {
        setHealth(data);
        setConnection("online");
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setHealth(null);
          setConnection("offline");
        }
      });

    return () => {
      controller.abort();
      requestController.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!loading) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setElapsed(
        Math.floor((Date.now() - requestStarted.current) / 1000),
      );
    }, 1000);

    return () => window.clearInterval(timer);
  }, [loading]);

  async function handleConnectionCheck() {
    setConnection("checking");

    try {
      const data = await getHealth();

      setHealth(data);
      setConnection("online");
      setError("");
    } catch {
      setHealth(null);
      setConnection("offline");
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (requestRunning.current) {
      return;
    }

    const trimmedQuestion = question.trim();

    if (trimmedQuestion.length < 3) {
      setError("Enter a question with at least 3 characters.");
      return;
    }

    if (connection !== "online") {
      setError("Connect to the library before asking a question.");
      return;
    }

    const controller = new AbortController();

    requestController.current = controller;
    requestRunning.current = true;
    requestStarted.current = Date.now();

    setLoading(true);
    setElapsed(0);
    setError("");
    setResult(null);

    try {
      const answer = await askQuestion(
        trimmedQuestion,
        controller.signal,
      );

      setResult(answer);
    } catch (requestError) {
      if (!controller.signal.aborted) {
        setError(
          requestError.message || "Something went wrong.",
        );

        if (requestError.status === 0) {
          setConnection("offline");
          setHealth(null);
        }
      }
    } finally {
      requestRunning.current = false;
      setLoading(false);
    }
  }

  function handleClear() {
    setQuestion("");
    setResult(null);
    setError("");
  }

  const connectionLabel = {
    checking: "Checking connection",
    online: "Library connected",
    offline: "Library offline",
  }[connection];

  const answerState = result
    ? ANSWER_STATES[result.status] || ANSWER_STATES.needs_review
    : null;

  const canSubmit =
    connection === "online" &&
    !loading &&
    question.trim().length >= 3;

  return (
    <div className="app-shell">
      <SpaceBackground />

      <header className="site-header">
        <div className="brand">
          <span className="brand-symbol" aria-hidden="true">
            ✦
          </span>

          <div>
            <span className="brand-name">AstroDocs</span>
            <span className="brand-caption">
              Astronomy, with evidence
            </span>
          </div>
        </div>

        <div className="connection-controls">
          <span className={`connection-state ${connection}`}>
            <span className="connection-dot" aria-hidden="true" />
            {connectionLabel}
          </span>

          <button
            className="text-button"
            type="button"
            onClick={handleConnectionCheck}
            disabled={connection === "checking" || loading}
          >
            Check connection
          </button>
        </div>
      </header>

      <main>
        <section className="hero" aria-labelledby="page-title">
          <div className="hero-copy">
            <p className="eyebrow">
              EXPLORE YOUR ASTRONOMY LIBRARY
            </p>

            <h1 id="page-title">
              A universe of questions.
              <br />
              <span>Answers with sources.</span>
            </h1>

            <p className="hero-description">
              Explore NASA documents, ask about space, and follow
              each citation back to its source passage.
            </p>

            <div className="library-stats">
              <span>
                <strong>{DOCUMENTS.length}</strong> reference PDFs
              </span>

              <span>
                <strong>{health?.indexed_chunks ?? "—"}</strong>{" "}
                indexed passages
              </span>
            </div>
          </div>

          <div className="space-art" aria-hidden="true" />
        </section>

        <div className="workspace">
          <div className="question-column">
            <section
              className="panel question-panel"
              aria-labelledby="question-heading"
            >
              <p className="eyebrow">START YOUR EXPLORATION</p>
              <h2 id="question-heading">Ask your library</h2>

              <form onSubmit={handleSubmit}>
                <label htmlFor="question">Your question</label>

                <textarea
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="What would you like to discover?"
                  rows={6}
                  maxLength={500}
                  disabled={loading}
                  aria-describedby="question-help"
                />

                <div className="input-help" id="question-help">
                  <span>Use 3–500 characters.</span>
                  <span className="character-count">
                    {question.length}/500
                  </span>
                </div>

                {connection === "offline" && (
                  <p className="notice warning" role="status">
                    Start the backend, then select Check connection.
                  </p>
                )}

                {error && (
                  <p className="notice error" role="alert">
                    {error}
                  </p>
                )}

                <div className="form-actions">
                  <button
                    className="primary-button"
                    type="submit"
                    disabled={!canSubmit}
                  >
                    {loading ? "Preparing answer…" : "Ask AstroDocs"}
                  </button>

                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleClear}
                    disabled={loading || (!question && !result && !error)}
                  >
                    Clear
                  </button>
                </div>
              </form>

              <div className="example-section">
                <h3>Try a question</h3>

                <div className="example-list">
                  {EXAMPLES.map((example) => (
                    <button
                      className="example-button"
                      type="button"
                      key={example}
                      disabled={loading}
                      onClick={() => {
                        setQuestion(example);
                        setError("");
                      }}
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <details className="panel library-panel">
              <summary>
                Reference library
                <span>{DOCUMENTS.length} PDFs</span>
              </summary>

              <ul className="document-list">
                {DOCUMENTS.map((document) => (
                  <li key={document.file}>
                    <strong>{document.title}</strong>
                    <span>{document.file}</span>
                  </li>
                ))}
              </ul>
            </details>
          </div>

          <section
            className="panel answer-panel"
            aria-labelledby="answer-heading"
            aria-busy={loading}
          >
            <div className="panel-heading">
              <div>
                <p className="eyebrow">FOLLOW THE EVIDENCE</p>
                <h2 id="answer-heading">Answer & sources</h2>
              </div>

              {result && (
                <span className={`answer-badge ${answerState.tone}`}>
                  {answerState.label}
                </span>
              )}
            </div>

            {loading && (
              <div className="loading-state" role="status">
                <span className="spinner" aria-hidden="true" />
                <h3>Exploring the library</h3>
                <p>
                  Searching the passages and preparing your answer.
                </p>
                <span className="elapsed-time" aria-hidden="true">
                  {elapsed}s elapsed
                </span>
              </div>
            )}

            {!loading && !result && (
              <div className="empty-state">
                <span className="empty-star" aria-hidden="true">
                  ✧
                </span>
                <h3>Your next discovery starts here</h3>
                <p>
                  Ask a question to see an answer and the passages
                  that support it.
                </p>
              </div>
            )}

            {!loading && result && (
              <div className="answer-content" aria-live="polite">
                <div className="answered-question">
                  <span>QUESTION</span>
                  <p>{result.question}</p>
                </div>

                <AnswerText
                  answer={result.answer}
                  sources={result.cited_sources}
                />

                {result.status === "verified_excerpt" && (
                  <p className="evidence-note">
                    Quoted text matches the cited sources. Read the
                    passages to assess relevance and completeness.
                  </p>
                )}

                {result.status === "needs_review" && (
                  <p className="notice warning">
                    This response needs review against the source
                    passages.
                  </p>
                )}

                {result.reached_token_budget && (
                  <p className="notice warning">
                    The response reached the generation limit.
                    Try a narrower question.
                  </p>
                )}

                <dl className="answer-metrics">
                  <div>
                    <dt>Total time</dt>
                    <dd>{formatSeconds(result.total_seconds)}</dd>
                  </div>
                  <div>
                    <dt>Retrieval</dt>
                    <dd>{formatSeconds(result.retrieval_seconds)}</dd>
                  </div>
                  <div>
                    <dt>Answer processing</dt>
                    <dd>{formatSeconds(result.generation_seconds)}</dd>
                  </div>
                </dl>

                <div className="sources-section">
                  <div className="sources-heading">
                    <h3>Cited sources</h3>
                    <span>{result.cited_sources.length}</span>
                  </div>

                  {result.cited_sources.length > 0 ? (
                    <div className="source-list">
                      {result.cited_sources.map((source) => (
                        <SourceCard
                          key={`${source.source_id}-${source.chunk_id}`}
                          source={source}
                        />
                      ))}
                    </div>
                  ) : (
                    <p className="no-sources">
                      No sources were cited for this answer.
                    </p>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="site-footer">
        <span>AstroDocs</span>
        <p>
          Answers reflect the supplied astronomy documents.
        </p>
      </footer>
    </div>
  );
}