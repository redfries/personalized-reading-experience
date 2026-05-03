import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  Upload,
  Sparkles,
  Download,
  Search,
  Settings2,
  Layers,
  FileText,
  ChevronDown,
  Info,
  UserRound,
  Save,
  Eye,
  RefreshCcw,
  CheckCircle2,
  FileCheck2,
  Loader2,
} from "lucide-react";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import { analyzePaper, fetchProfiles, previewProfile, saveProfile } from "./api";
import "./styles.css";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const TOPICS = [
  "Machine Learning",
  "Natural Language Processing",
  "Recommender Systems",
  "Information Retrieval",
  "Human-Computer Interaction",
  "Education AI",
  "Healthcare AI",
  "Computer Vision",
  "Explainable AI",
  "Semantic Search",
  "Academic Reading",
  "Personalization",
];

const MAX_RENDERED_PAGES = 25;

const MODE_HELP = {
  Concise: "Careful mode: fewer, stronger, mostly sentence-level highlights.",
  Standard: "Balanced mode: broader than Concise and can include paragraph context.",
  Expanded: "Broad scan mode: widest search. If the paper is weakly related, it looks for possible bridge points and marks them as weaker signals.",
};

function Pill({ children, tone = "default" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function TopBar({ activeView, setActiveView }) {
  return (
    <header className="topbar topbarNoAction">
      <div className="brand">
        <div className="brandIcon"><BookOpen size={20} /></div>
        <div>
          <div className="brandTitle">Personalized Reading Assistant</div>
          <div className="brandSub">Dynamic explanations + grounded Gemini note</div>
        </div>
      </div>
      <nav className="navTabs">
        <button className={activeView === "profile" ? "navActive" : ""} onClick={() => setActiveView("profile")}>Profile Builder</button>
        <button className={activeView === "reader" ? "navActive" : ""} onClick={() => setActiveView("reader")}>Reader</button>
      </nav>
      <div className="topbarSpacer" />
    </header>
  );
}

function ProfileBuilder({ onSavedProfile }) {
  const [profileName, setProfileName] = useState("AI Reading Assistant Profile");
  const [selectedTopics, setSelectedTopics] = useState(["Recommender Systems", "Information Retrieval"]);
  const [keywords, setKeywords] = useState("personalization, semantic search, academic papers, user profile");
  const [researchStatement, setResearchStatement] = useState("");
  const [seedPapers, setSeedPapers] = useState([]);
  const [preview, setPreview] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const seedPaperNames = Array.from(seedPapers).map((file) => file.name);

  function toggleTopic(topic) {
    setSelectedTopics((current) => current.includes(topic) ? current.filter((item) => item !== topic) : [...current, topic]);
  }

  function payload() {
    return {
      profile_name: profileName,
      selected_topics: selectedTopics,
      keywords,
      research_statement: researchStatement,
      seed_papers: seedPapers,
    };
  }

  async function handlePreview() {
    setBusy(true);
    setMessage("");
    try {
      const data = await previewProfile(payload());
      setPreview(data);
    } catch (err) {
      setMessage(err.message || "Preview failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleSave() {
    setBusy(true);
    setMessage("");
    try {
      const data = await saveProfile(payload());
      setPreview(data.profile);
      setMessage(data.message);
      if (data.ok) onSavedProfile?.(data.profile);
    } catch (err) {
      setMessage(err.message || "Save failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="profilePage">
      <section className="profileHero card">
        <div>
          <Pill tone="blue">v4.3.5 Dynamic Explanations</Pill>
          <h1>Create a reusable research profile</h1>
          <p>
            Build a profile from topics, keywords, optional research statement, and seed PDFs.
            If a target paper has too few strong highlights, the reader expands the search and shows weaker bridge points honestly.
          </p>
        </div>
        <div className="heroStat">
          <strong>{preview?.profile_strength || "Not previewed"}</strong>
          <span>profile strength</span>
        </div>
      </section>

      <div className="profileGrid">
        <section className="card">
          <div className="sectionTitle"><UserRound size={16} /> Profile sources</div>

          <label className="label">Profile name</label>
          <input className="input" value={profileName} onChange={(e) => setProfileName(e.target.value)} />

          <label className="label">Research areas</label>
          <div className="topicGrid">
            {TOPICS.map((topic) => (
              <button key={topic} className={`topicChip ${selectedTopics.includes(topic) ? "topicChipActive" : ""}`} onClick={() => toggleTopic(topic)}>
                {topic}
              </button>
            ))}
          </div>

          <label className="label">Keywords</label>
          <textarea className="textarea smallTextArea" value={keywords} onChange={(e) => setKeywords(e.target.value)} placeholder="personalization, semantic search, paper recommendation" />

          <label className="label">Research statement, optional</label>
          <textarea className="textarea" value={researchStatement} onChange={(e) => setResearchStatement(e.target.value)} placeholder="Paste a short project idea, thesis paragraph, or advisor topic note." />

          <label className="label">Seed papers, optional</label>
          <label className="fileBox">
            <Upload size={18} />
            <span>{seedPapers.length ? `${seedPapers.length} file(s) selected` : "Choose seed PDFs"}</span>
            <input type="file" accept="application/pdf" multiple onChange={(e) => setSeedPapers(Array.from(e.target.files || []).slice(0, 3))} />
          </label>

          {seedPaperNames.length > 0 && <div className="seedList">{seedPaperNames.map((name) => <Pill key={name}>{name}</Pill>)}</div>}

          <div className="buttonRow">
            <button className="secondaryBtn" onClick={handlePreview} disabled={busy}><Eye size={16} /> Preview profile</button>
            <button className="primaryBtn" onClick={handleSave} disabled={busy}><Save size={16} /> Save real profile</button>
          </div>
          {message && <div className="notice">{message}</div>}
        </section>

        <section className="card previewCard">
          <div className="sectionTitle"><Sparkles size={16} /> Profile preview</div>
          {preview ? (
            <>
              <div className="previewTop">
                <div>
                  <h2>{preview.profile_name}</h2>
                  <div className="pillRow">
                    <Pill tone={preview.profile_strength === "Very strong" ? "dark" : "blue"}>{preview.profile_strength}</Pill>
                    <Pill>{preview.source_count} source(s)</Pill>
                  </div>
                </div>
                <CheckCircle2 size={28} className="successIcon" />
              </div>
              <div className="miniBox">
                <div className="miniTitle">Sources used</div>
                <div className="pillRow">
                  {preview.sources_used?.length ? preview.sources_used.map((source) => <Pill key={source}>{source}</Pill>) : <span className="muted">No sources yet.</span>}
                </div>
              </div>
              <div className="textBlock">
                <div className="miniTitle">Combined profile text</div>
                <p>{preview.combined_profile_text}</p>
              </div>
            </>
          ) : (
            <div className="emptyState">
              <Sparkles size={28} />
              <h2>No preview yet</h2>
              <p>Click Preview profile to see how the system will represent the user interest profile.</p>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}


function ModeButton({ mode, density, setDensity }) {
  return (
    <button className={`modeBtn ${density === mode ? "modeActive" : ""}`} onClick={() => setDensity(mode)}>
      <span>{mode}</span>
      <span className="infoDot" title={MODE_HELP[mode]} onClick={(event) => event.stopPropagation()}>i</span>
    </button>
  );
}

function MatchCard({ analysis }) {
  if (!analysis?.profile_match) return null;
  const match = analysis.profile_match;
  const labelClass = match.label?.toLowerCase().replaceAll(" ", "-") || "unknown";

  return (
    <section className={`card matchCard match-${labelClass}`}>
      <div className="sectionTitle"><Sparkles size={16} /> Profile-paper match</div>
      <div className="matchTop">
        <strong>{match.label}</strong>
        <span>{analysis.analysis_mode_used || "Standard"}</span>
      </div>
      <p>{match.message}</p>
      <div className="matchStats">
        <div><span>Top score</span><strong>{Number(match.top_score || 0).toFixed(4)}</strong></div>
        <div><span>Top 5 avg</span><strong>{Number(match.top_5_average || 0).toFixed(4)}</strong></div>
        <div><span>Candidates</span><strong>{match.candidate_count || 0}</strong></div>
      </div>
    </section>
  );
}



function LLMNoteCard({ note }) {
  if (!note) {
    return (
      <section className="card llmNoteCard llm-disabled">
        <div className="sectionTitle"><Sparkles size={16} /> AI relevance note</div>
        <p className="llmSummary">Run analysis to generate or check the Gemini relevance note.</p>
      </section>
    );
  }

  const isOk = note.status === "ok";
  const isDisabled = note.status === "disabled";

  return (
    <section className={`card llmNoteCard ${isOk ? "llm-ok" : isDisabled ? "llm-disabled" : "llm-error"}`}>
      <div className="sectionTitle"><Sparkles size={16} /> AI relevance note</div>
      <div className="llmStatusRow">
        <strong>{note.overall_match || note.title || "AI relevance note"}</strong>
        <span>{note.provider || "gemini"}{note.model ? ` · ${note.model}` : ""}</span>
      </div>

      {note.summary && <p className="llmSummary">{note.summary}</p>}

      {note.why_relevant?.length > 0 && (
        <div className="miniBox compactMini">
          <div className="miniTitle">Why this may matter</div>
          <div className="evidenceList">
            {note.why_relevant.map((item, index) => (
              <div className="evidenceItem" key={`${item.point}-${index}`}>
                <strong>{item.point}</strong>
                {item.section && <span>{item.section}</span>}
                {item.pdf_evidence && <blockquote>{item.pdf_evidence}</blockquote>}
              </div>
            ))}
          </div>
        </div>
      )}

      {note.weak_or_missing_connections?.length > 0 && (
        <div className="miniBox compactMini">
          <div className="miniTitle">Weak or missing connections</div>
          <ul className="bridgeList">
            {note.weak_or_missing_connections.map((point, index) => <li key={`${point}-${index}`}>{point}</li>)}
          </ul>
        </div>
      )}

      {note.reading_advice && (
        <div className="miniBox compactMini">
          <div className="miniTitle">Reading advice</div>
          <p>{note.reading_advice}</p>
        </div>
      )}

      {note.sections_to_focus?.length > 0 && (
        <div className="miniBox compactMini">
          <div className="miniTitle">Sections to focus</div>
          <div className="sectionFocusList">
            {note.sections_to_focus.map((item, index) => (
              <div key={`${item.section}-${index}`}>
                <strong>{item.section}</strong>
                <p>{item.reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {note.bridge_points?.length > 0 && (
        <div className="miniBox compactMini">
          <div className="miniTitle">Bridge points</div>
          <ul className="bridgeList">
            {note.bridge_points.map((point, index) => <li key={`${point}-${index}`}>{point}</li>)}
          </ul>
        </div>
      )}

      {note.caution && <div className="llmCaution">{note.caution}</div>}
    </section>
  );
}


function Controls({ profiles, selectedProfileId, setSelectedProfileId, refreshProfiles, density, setDensity, file, setFile, analysis, onAnalyze, loading }) {
  const selectedProfile = profiles.find((p) => p.profile_id === selectedProfileId);
  return (
    <aside className="leftPanel">
      <section className="card">
        <div className="sectionTitle"><FileText size={16} /> Paper session</div>
        <label className="label">Saved profile</label>
        <div className="selectRow">
          <select className="select" value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}>
            <option value="">Select a saved profile</option>
            {profiles.map((profile) => <option value={profile.profile_id} key={profile.profile_id}>{profile.profile_name}</option>)}
          </select>
          <button className="iconBtn" onClick={refreshProfiles} title="Refresh profiles"><RefreshCcw size={16} /></button>
        </div>
        {selectedProfile ? (
          <div className="selectedProfileBox">
            <strong>{selectedProfile.profile_strength}</strong>
            <span>{selectedProfile.sources_used.join(", ")}</span>
            {selectedProfile.model_name && <span>{selectedProfile.model_name}</span>}
          </div>
        ) : (
          <p className="helperMini">Create and save a profile first, then select it here.</p>
        )}
        <label className="label">Upload target paper</label>
        <label className="fileBox">
          <Upload size={18} />
          <span>{file ? file.name : "Choose PDF"}</span>
          <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        </label>
        <button className="primaryBtn fullWidthBtn" onClick={onAnalyze} disabled={loading}>
          <Sparkles size={16} />
          {loading ? "Analyzing..." : "Analyze paper"}
        </button>
      </section>
      <section className="card">
        <div className="sectionTitle"><Settings2 size={16} /> Reading mode</div>
        <div className="modeGroup">
          {["Concise", "Standard", "Expanded"].map((mode) => (
            <ModeButton key={mode} mode={mode} density={density} setDensity={setDensity} />
          ))}
        </div>
      </section>
      <MatchCard analysis={analysis} />
      <LLMNoteCard note={analysis?.llm_note} />
      <section className="card">
        <div className="sectionTitle"><Info size={16} /> Summary</div>
        {analysis ? (
          <>
            <div className="summaryGrid">
              <div><span>Mode</span><strong>{analysis.summary.mode}</strong></div>
              <div><span>Requested</span><strong>{analysis.summary.density}</strong></div>
              <div><span>Used</span><strong>{analysis.analysis_mode_used || analysis.summary.density}</strong></div>
              <div><span>Highlights</span><strong>{analysis.summary.visible_highlights}</strong></div>
              <div><span>PDF matches</span><strong>{analysis.summary.pdf_matches}</strong></div>
              {analysis.summary?.mode_config && <div><span>Threshold</span><strong>{Number(analysis.summary.mode_config.threshold).toFixed(2)}</strong></div>}
              {analysis.summary?.mode_config && <div><span>Top limit</span><strong>{analysis.summary.mode_config.top_k}</strong></div>}
            </div>
            {analysis.summary?.warnings?.map((warning) => <div className="warningMini" key={warning}>{warning}</div>)}
          </>
        ) : <p className="muted">Run analysis to see summary.</p>}
      </section>
    </aside>
  );
}

function HighlightList({ highlights, activeId, setActiveId }) {
  if (!highlights?.length) {
    return (
      <section className="card highlightList">
        <div className="sectionTitle"><Search size={16} /> Highlights</div>
        <p className="muted">No highlights yet.</p>
      </section>
    );
  }

  return (
    <section className="card highlightList">
      <div className="sectionTitle"><Search size={16} /> Highlights</div>
      {highlights.map((h, index) => (
        <button key={h.highlight_id} className={`highlightItem ${activeId === h.highlight_id ? "highlightItemActive" : ""}`} onClick={() => setActiveId(h.highlight_id)}>
          <div className="highlightItemTop"><strong>Highlight {index + 1}</strong><span>p. {h.page}</span></div>
          <p>{h.highlighted_sentence}</p>
          <div className="highlightBadges">
            {h.low_confidence && <span className="weakBadge">Weak bridge</span>}
            {!h.pdf_match && <span className="noPdfMatch">No PDF box</span>}
          </div>
        </button>
      ))}
    </section>
  );
}

function PdfCanvasPage({ pdf, pageNumber, highlights, activeId, setActiveId }) {
  const canvasRef = useRef(null);
  const pageShellRef = useRef(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let cancelled = false;
    let renderTask = null;

    async function renderPage() {
      try {
        setStatus("loading");
        const page = await pdf.getPage(pageNumber);
        if (cancelled) return;

        const shellWidth = pageShellRef.current?.clientWidth || 760;
        const baseViewport = page.getViewport({ scale: 1 });
        const scale = Math.min(shellWidth / baseViewport.width, 1.7);
        const viewport = page.getViewport({ scale });

        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");
        const outputScale = window.devicePixelRatio || 1;

        canvas.width = Math.floor(viewport.width * outputScale);
        canvas.height = Math.floor(viewport.height * outputScale);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;

        context.setTransform(outputScale, 0, 0, outputScale, 0, 0);
        renderTask = page.render({ canvasContext: context, viewport });
        await renderTask.promise;

        if (!cancelled) setStatus("ready");
      } catch (err) {
        if (!cancelled) {
          console.error(err);
          setStatus("error");
        }
      }
    }

    renderPage();

    return () => {
      cancelled = true;
      if (renderTask) {
        try { renderTask.cancel(); } catch {}
      }
    };
  }, [pdf, pageNumber]);

  const pageHighlights = highlights.filter((h) => h.page === pageNumber && Array.isArray(h.rects) && h.rects.length > 0);

  return (
    <div className="pdfPageShell" ref={pageShellRef}>
      <div className="pdfPageLabel">Page {pageNumber}</div>
      <div className="pdfCanvasWrap">
        <canvas ref={canvasRef} className="pdfCanvas" />
        {status === "loading" && <div className="pdfPageLoading"><Loader2 size={18} className="spin" /> Rendering page...</div>}
        {status === "error" && <div className="pdfPageLoading">Could not render this page.</div>}
        <div className="pdfOverlay">
          {pageHighlights.flatMap((h) => h.rects.map((r, idx) => (
            <button
              key={`${h.highlight_id}-${idx}`}
              className={`pdfRealHighlight ${h.low_confidence ? "pdfWeakHighlight" : ""} ${activeId === h.highlight_id ? "pdfRealHighlightActive" : ""}`}
              style={{ left: `${r.x}%`, top: `${r.y}%`, width: `${r.width}%`, height: `${r.height}%` }}
              onClick={() => setActiveId(h.highlight_id)}
              title={h.highlighted_sentence}
            />
          )))}
        </div>
      </div>
    </div>
  );
}

function PdfReader({ analysis, activeId, setActiveId }) {
  const [pdf, setPdf] = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setPdf(null);
    setPageCount(0);
    setError("");

    if (!analysis?.pdf_url) return;

    const task = pdfjsLib.getDocument(analysis.pdf_url);
    task.promise
      .then((loadedPdf) => {
        if (cancelled) return;
        setPdf(loadedPdf);
        setPageCount(loadedPdf.numPages);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error(err);
        setError("Could not load the PDF for inline viewing.");
      });

    return () => {
      cancelled = true;
      try { task.destroy(); } catch {}
    };
  }, [analysis?.pdf_url]);

  if (!analysis) {
    return (
      <div className="paperCanvas">
        <div className="emptyReaderState">
          <FileCheck2 size={36} />
          <h2>No paper analyzed yet</h2>
          <p>Select a saved profile, upload a target PDF, and click Analyze paper.</p>
        </div>
      </div>
    );
  }

  if (!analysis?.pdf_url) {
    return (
      <div className="paperCanvas">
        <div className="emptyReaderState">
          <FileCheck2 size={36} />
          <h2>No PDF attached</h2>
          <p>This analysis does not have an uploaded PDF file.</p>
        </div>
      </div>
    );
  }

  if (error) return <div className="paperCanvas"><div className="errorCard">{error}</div></div>;
  if (!pdf) return <div className="paperCanvas"><div className="pdfLoading"><Loader2 size={20} className="spin" /> Loading PDF...</div></div>;

  const pagesToRender = Array.from({ length: Math.min(pageCount, MAX_RENDERED_PAGES) }, (_, i) => i + 1);

  return (
    <div className="paperCanvas pdfCanvasArea">
      {pageCount > MAX_RENDERED_PAGES && (
        <div className="mockNotice">Showing the first {MAX_RENDERED_PAGES} pages for performance. The analysis still ranked text from the extracted document.</div>
      )}
      {analysis?.analysis_mode_used === "Expanded rescue" && (
        <div className="rescueNotice">
          This paper appears weakly related to your profile. I expanded the search to find possible bridge points, but these are weaker signals.
        </div>
      )}
      {analysis?.summary?.filters?.length > 0 && (
        <div className="filterNotice">Quality filters active: references, table-like blocks, duplicate candidates, overlapping rectangles, and adaptive rescue.</div>
      )}
      {pagesToRender.map((pageNumber) => (
        <PdfCanvasPage key={pageNumber} pdf={pdf} pageNumber={pageNumber} highlights={analysis?.highlights || []} activeId={activeId} setActiveId={setActiveId} />
      ))}
    </div>
  );
}

function Reader({ analysis, activeId, setActiveId }) {
  const hasRealPdf = Boolean(analysis?.pdf_url);
  return (
    <main className="reader">
      <div className="readerHeader">
        <div>
          <h1>{hasRealPdf ? "Real PDF highlight view" : "Highlighted paper view"}</h1>
          <p>{hasRealPdf ? "The uploaded PDF is rendered in-browser. Click a yellow highlight to inspect why it was selected." : "Upload and analyze a PDF to see real personalized highlights."}</p>
        </div>
        {analysis?.pdf_url ? (
          <a className="secondaryBtn" href={analysis.pdf_url} target="_blank" rel="noreferrer"><Download size={16} /> Open PDF</a>
        ) : (
          <button className="secondaryBtn" disabled><FileCheck2 size={16} /> Waiting for analysis</button>
        )}
      </div>
      <PdfReader analysis={analysis} activeId={activeId} setActiveId={setActiveId} />
    </main>
  );
}

function InsightPanel({ highlight }) {
  if (!highlight) return <aside className="rightPanel"><section className="card"><div className="sectionTitle"><Layers size={16} /> Highlight insight</div><p className="muted">Select a highlight to see explanation.</p></section></aside>;
  return (
    <aside className="rightPanel">
      <section className="card insightCard">
        <div className="sectionTitle"><Layers size={16} /> Highlight insight</div>
        <div className="pillRow"><Pill tone="dark">{highlight.section}</Pill><Pill tone="amber">{highlight.tag}</Pill><Pill tone="blue">{highlight.scope}</Pill>{highlight.low_confidence && <Pill tone="warning">Weak bridge</Pill>}</div>
        <h2>Why this relates to your work</h2>
        <p className="explanation">{highlight.explanation}</p>
        <div className="miniBox"><div className="miniTitle">Matched profile sources</div><div className="pillRow">{highlight.matched_sources.map((source) => <Pill key={source}>{source}</Pill>)}</div></div>
        <div className="textBlock"><div className="miniTitle">Highlighted sentence</div><p>{highlight.highlighted_sentence}</p></div>
        <details className="detailsBox"><summary>Paragraph context <ChevronDown size={14} /></summary><p>{highlight.paragraph_context}</p></details>
        <details className="detailsBox"><summary>Advanced details <ChevronDown size={14} /></summary><div className="advancedGrid"><div><span>Page</span><strong>{highlight.page}</strong></div><div><span>Label</span><strong>{highlight.label}</strong></div><div><span>Sentence score</span><strong>{Number(highlight.sentence_score).toFixed(4)}</strong></div><div><span>Paragraph score</span><strong>{Number(highlight.paragraph_score).toFixed(4)}</strong></div><div><span>PDF match</span><strong>{highlight.pdf_match ? "Yes" : "No"}</strong></div><div><span>Confidence</span><strong>{highlight.low_confidence ? "Weak bridge" : "Normal"}</strong></div></div></details>
      </section>
    </aside>
  );
}

function ReaderPage({ profiles, refreshProfiles }) {
  const [analysis, setAnalysis] = useState(null);
  const [activeId, setActiveId] = useState(null);
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [density, setDensity] = useState("Standard");
  const [file, setFile] = useState(null);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [errorLocal, setErrorLocal] = useState("");

  const activeHighlight = useMemo(() => analysis?.highlights?.find((h) => h.highlight_id === activeId) || null, [analysis, activeId]);
  const selectedProfile = profiles.find((p) => p.profile_id === selectedProfileId);
  const profileName = selectedProfile?.profile_name || "AI Reading Assistant Profile";

  async function handleAnalyze() {
    try {
      setLoadingLocal(true);
      setErrorLocal("");
      if (!selectedProfileId) throw new Error("Please select a saved profile before analyzing.");
      if (!file) throw new Error("Please upload a target PDF before analyzing.");
      const data = await analyzePaper({ profileName, profileId: selectedProfileId, density, file });
      setAnalysis(data);
      setActiveId(data.highlights?.[0]?.highlight_id || null);
    } catch (err) {
      setErrorLocal(err.message || "Analysis failed");
    } finally {
      setLoadingLocal(false);
    }
  }

  return (
    <>
      {errorLocal && <div className="errorBanner">{errorLocal}</div>}
      <div className="layout">
        <div className="leftStack">
          <Controls profiles={profiles} selectedProfileId={selectedProfileId} setSelectedProfileId={setSelectedProfileId} refreshProfiles={refreshProfiles} density={density} setDensity={setDensity} file={file} setFile={setFile} analysis={analysis} onAnalyze={handleAnalyze} loading={loadingLocal} />
          <HighlightList highlights={analysis?.highlights || []} activeId={activeId} setActiveId={setActiveId} />
        </div>
        <Reader analysis={analysis} activeId={activeId} setActiveId={setActiveId} />
        <InsightPanel highlight={activeHighlight} />
      </div>
    </>
  );
}

function App() {
  const [activeView, setActiveView] = useState("profile");
  const [profiles, setProfiles] = useState([]);
  const [globalMessage, setGlobalMessage] = useState("");

  async function refreshProfiles() {
    try {
      const data = await fetchProfiles();
      setProfiles(data.profiles || []);
    } catch (err) {
      setGlobalMessage(err.message || "Failed to refresh profiles");
    }
  }

  useEffect(() => { refreshProfiles(); }, []);

  function handleSavedProfile() {
    refreshProfiles();
    setGlobalMessage("Real profile saved. You can now use it in the Reader.");
  }

  return (
    <div className="appShell">
      <TopBar activeView={activeView} setActiveView={setActiveView} />
      {globalMessage && <div className="noticeBanner">{globalMessage}</div>}
      {activeView === "profile" ? <ProfileBuilder onSavedProfile={handleSavedProfile} /> : <ReaderPage profiles={profiles} refreshProfiles={refreshProfiles} />}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
