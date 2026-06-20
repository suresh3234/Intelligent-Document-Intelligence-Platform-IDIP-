import { useState, useEffect, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  FileText,
  AlertCircle,
  RefreshCw,
  Trash2,
  Eye,
  ChevronLeft,
  ChevronRight,
  Filter,
  Calendar,
  Info,
  Clock,
  Check,
  Loader2,
  Sparkles,
  Send,
  MessageSquare,
  Copy,
  User,
  Search,
  ExternalLink,
  ShieldAlert,
  Bot
} from 'lucide-react';

// JWT Generation using Browser Web Crypto API
async function generateJWT(apiKey = "anonymous", secret = "idip_secret_key_1234567890"): Promise<string> {
  const header = { alg: "HS256", typ: "JWT" };
  const payload = {
    sub: apiKey,
    api_key: apiKey,
    exp: Math.floor(Date.now() / 1000) + 3600
  };

  const base64UrlEncode = (obj: any): string => {
    const str = JSON.stringify(obj);
    const base64 = btoa(unescape(encodeURIComponent(str)));
    return base64.replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");
  };

  const headerB64 = base64UrlEncode(header);
  const payloadB64 = base64UrlEncode(payload);
  const signingInput = `${headerB64}.${payloadB64}`;

  const enc = new TextEncoder();
  const keyData = enc.encode(secret);
  const messageData = enc.encode(signingInput);

  const key = await window.crypto.subtle.importKey(
    "raw",
    keyData,
    { name: "HMAC", hash: { name: "SHA-256" } },
    false,
    ["sign"]
  );

  const signature = await window.crypto.subtle.sign(
    "HMAC",
    key,
    messageData
  );

  const signatureB64 = btoa(String.fromCharCode(...new Uint8Array(signature)))
    .replace(/=/g, "").replace(/\+/g, "-").replace(/\//g, "_");

  return `${signingInput}.${signatureB64}`;
}

interface DocumentListItem {
  doc_id: string;
  filename: string;
  source_type: string;
  status: string;
  doc_type?: string;
  ingestion_ts: string;
}

interface PipelineStep {
  stepName: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  elapsed: number;
  error: string | null;
}

interface ActivePipeline {
  docId: string;
  filename: string;
  steps: PipelineStep[];
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHrs = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHrs / 24);

  if (isNaN(date.getTime())) return "Unknown";
  if (diffSecs < 60) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHrs < 24) return `${diffHrs}h ago`;
  return `${diffDays}d ago`;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const limit = 20;

  // Filters
  const [statusFilter, setStatusFilter] = useState('');
  const [sourceTypeFilter, setSourceTypeFilter] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Selection
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set());

  // Upload progress
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadingFile, setUploadingFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Active SSE status pipeline
  const [activePipeline, setActivePipeline] = useState<ActivePipeline | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Modals / Overlays
  const [viewingDocDetails, setViewingDocDetails] = useState<any | null>(null);
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null);
  const [showBulkDeleteConfirm, setShowBulkDeleteConfirm] = useState(false);

  // System Uptime & Status Info
  const [systemHealth, setSystemHealth] = useState<'healthy' | 'degraded' | 'offline'>('healthy');

  // Chat / Q&A States
  const [activeTab, setActiveTab] = useState<'documents' | 'chat'>('documents');
  const [chatQuery, setChatQuery] = useState('');
  const [chatScopeDocId, setChatScopeDocId] = useState(''); // Empty string is all documents
  
  interface ChatMessage {
    id: string;
    sender: 'user' | 'assistant';
    text: string;
    streaming?: boolean;
    citations?: {
      doc_id: string;
      doc_id_short: string;
      source_uri: string;
      text_snippet: string;
    }[];
    confidence?: number;
    lowConfidence?: boolean;
  }
  
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: "Hello! I am your IDIP AI Assistant. Ask me questions about your uploaded documents, and I'll retrieve the relevant chunks to answer you, complete with citation tags and confidence metrics!"
    }
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll chat window when history updates
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  // Fetch list of documents
  const fetchDocuments = useCallback(async () => {
    try {
      const token = await generateJWT();
      let url = `/v1/documents?page=${page}&limit=${limit}`;
      if (statusFilter) url += `&status=${statusFilter}`;
      if (sourceTypeFilter) url += `&source_type=${sourceTypeFilter}`;
      if (startDate) url += `&start_date=${new Date(startDate).toISOString()}`;
      if (endDate) url += `&end_date=${new Date(endDate).toISOString()}`;

      const res = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents);
        setTotal(data.total);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  }, [page, statusFilter, sourceTypeFilter, startDate, endDate]);

  // Fetch all completed documents for chat dropdown (unpaginated)
  const [chatDocuments, setChatDocuments] = useState<DocumentListItem[]>([]);

  const fetchChatDocuments = useCallback(async () => {
    try {
      const token = await generateJWT();
      const res = await fetch('/v1/documents?page=1&limit=1000&status=completed', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        const data = await res.json();
        setChatDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Failed to fetch completed documents for chat dropdown:', err);
    }
  }, []);

  useEffect(() => {
    fetchChatDocuments();
  }, [fetchChatDocuments]);

  // Load documents and check health on mount / change
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/v1/health');
        if (res.ok) {
          const data = await res.json();
          setSystemHealth(data.status === 'healthy' ? 'healthy' : 'degraded');
        } else {
          setSystemHealth('offline');
        }
      } catch {
        setSystemHealth('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Set up SSE Pipeline status tracker
  const startPipelineMonitoring = (docId: string, filename: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const stepsList = ["Received", "Validating", "Chunking", "Embedding", "Indexing", "Complete"];
    setActivePipeline({
      docId,
      filename,
      steps: stepsList.map(step => ({
        stepName: step,
        status: 'pending',
        elapsed: 0,
        error: null
      }))
    });

    const eventSource = new EventSource(`/v1/documents/${docId}/status`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setActivePipeline(prev => {
          if (!prev || prev.docId !== docId) return prev;
          const updatedSteps = prev.steps.map(step => {
            if (step.stepName === data.step) {
              return {
                ...step,
                status: data.status,
                elapsed: data.elapsed_time,
                error: data.error_message
              };
            }
            // Auto complete previous steps if a subsequent one is completed
            const stepsIdx = stepsList.indexOf(step.stepName);
            const dataIdx = stepsList.indexOf(data.step);
            if (stepsIdx < dataIdx && data.status === 'completed') {
              return { ...step, status: 'completed' as const };
            }
            return step;
          });

          return {
            ...prev,
            steps: updatedSteps
          };
        });

        // Close when complete or failed, refresh catalog
        if ((data.step === "Complete" && data.status === "completed") || data.status === "failed") {
          eventSource.close();
          fetchDocuments();
          fetchChatDocuments();
        }
      } catch (err) {
        console.error("Error parsing SSE event:", err);
      }
    };

    eventSource.onerror = () => {
      // Gracefully handle or retry
      eventSource.close();
    };
  };

  // Upload file drop handler
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    const file = acceptedFiles[0];

    // File limit check (50MB)
    if (file.size > 50 * 1024 * 1024) {
      setUploadError("File size exceeds the 50MB limit.");
      return;
    }

    setUploadingFile(file);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const token = await generateJWT();
      const metadata = JSON.stringify({
        source_uri: `upload://${file.name}`,
        source_type: file.name.split('.').pop()?.toLowerCase() || 'pdf'
      });

      // Perform multipart upload using XMLHttpRequest to track progress
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/v1/documents/ingest");
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(pct);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          const resp = JSON.parse(xhr.responseText);
          // Start SSE tracking
          startPipelineMonitoring(resp.doc_id, file.name);
          setUploadingFile(null);
          setUploadProgress(null);
          fetchDocuments();
        } else {
          const errResp = JSON.parse(xhr.responseText || '{}');
          setUploadError(errResp.message || `Upload failed with status ${xhr.status}`);
          setUploadingFile(null);
          setUploadProgress(null);
        }
      };

      xhr.onerror = () => {
        setUploadError("Network connection error occurred during upload.");
        setUploadingFile(null);
        setUploadProgress(null);
      };

      const formData = new FormData();
      formData.append("file", file);
      formData.append("metadata", metadata);
      xhr.send(formData);

    } catch (err: any) {
      setUploadError(err.message || "An unexpected error occurred.");
      setUploadingFile(null);
      setUploadProgress(null);
    }
  }, [fetchDocuments]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg']
    },
    multiple: false
  });

  const handleSendQuery = async (forcedQuery?: string) => {
    const queryToSend = forcedQuery !== undefined ? forcedQuery : chatQuery;
    if (!queryToSend.trim() || isGenerating) return;

    if (forcedQuery === undefined) {
      setChatQuery('');
    }

    const userMsgId = `msg-${Date.now()}-user`;
    const aiMsgId = `msg-${Date.now()}-ai`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      sender: 'user',
      text: queryToSend
    };
    const aiMsg: ChatMessage = {
      id: aiMsgId,
      sender: 'assistant',
      text: '',
      streaming: true
    };

    setChatHistory(prev => [...prev, userMsg, aiMsg]);
    setIsGenerating(true);

    try {
      const token = await generateJWT();
      const payload: any = {
        query: queryToSend,
        top_k: 5
      };
      if (chatScopeDocId) {
        payload.filters = { doc_id: chatScopeDocId };
      }

      const response = await fetch('/v1/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Streaming response body is null.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let answerText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const cleanLine = line.trim();
          if (!cleanLine.startsWith('data: ')) continue;
          const dataStr = cleanLine.slice(6);
          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.token !== undefined) {
              answerText += parsed.token;
              setChatHistory(prev => prev.map(msg => {
                if (msg.id === aiMsgId) {
                  return { ...msg, text: answerText };
                }
                return msg;
              }));
            } else if (parsed.citations !== undefined) {
              setChatHistory(prev => prev.map(msg => {
                if (msg.id === aiMsgId) {
                  return {
                    ...msg,
                    citations: parsed.citations,
                    confidence: parsed.confidence,
                    lowConfidence: parsed.low_confidence
                  };
                }
                return msg;
              }));
            }
          } catch (e) {
            console.error('SSE JSON parse error:', e);
          }
        }
      }

      setChatHistory(prev => prev.map(msg => {
        if (msg.id === aiMsgId) {
          return { ...msg, streaming: false };
        }
        return msg;
      }));

    } catch (error: any) {
      console.error('Error executing query stream:', error);
      setChatHistory(prev => prev.map(msg => {
        if (msg.id === aiMsgId) {
          return {
            ...msg,
            streaming: false,
            text: `Error generating answer: ${error.message || 'Server connection failed.'}`
          };
        }
        return msg;
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopyAnswer = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  // Action: Reprocess Document
  const handleReprocess = async (docId: string, filename: string) => {
    try {
      const token = await generateJWT();
      const res = await fetch(`/v1/documents/${docId}/reprocess`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        startPipelineMonitoring(docId, filename);
        fetchDocuments();
      } else {
        alert("Failed to queue document for reprocessing.");
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Action: View Details
  const handleViewDetails = async (docId: string) => {
    try {
      const token = await generateJWT();
      const res = await fetch(`/v1/documents/${docId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        const data = await res.json();
        setViewingDocDetails(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Action: Delete Document
  const handleDeleteConfirm = async () => {
    if (!deletingDocId) return;
    try {
      const token = await generateJWT();
      const res = await fetch(`/v1/documents/${deletingDocId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (res.ok) {
        setDocuments(prev => prev.filter(doc => doc.doc_id !== deletingDocId));
        setSelectedDocIds(prev => {
          const next = new Set(prev);
          next.delete(deletingDocId);
          return next;
        });
        setDeletingDocId(null);
        fetchDocuments();
        fetchChatDocuments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Action: Bulk Delete
  const handleBulkDeleteConfirm = async () => {
    if (selectedDocIds.size === 0) return;
    try {
      const token = await generateJWT();
      const res = await fetch('/v1/documents/bulk-delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ doc_ids: Array.from(selectedDocIds) })
      });
      if (res.ok) {
        setSelectedDocIds(new Set());
        setShowBulkDeleteConfirm(false);
        fetchDocuments();
        fetchChatDocuments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Master selection toggle
  const toggleSelectAll = () => {
    if (selectedDocIds.size === documents.length) {
      setSelectedDocIds(new Set());
    } else {
      setSelectedDocIds(new Set(documents.map(d => d.doc_id)));
    }
  };

  const toggleSelectDoc = (docId: string) => {
    setSelectedDocIds(prev => {
      const next = new Set(prev);
      if (next.has(docId)) {
        next.delete(docId);
      } else {
        next.add(docId);
      }
      return next;
    });
  };

  // Relative status styles
  const getStatusBadgeClass = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/30';
      case 'processing':
        return 'bg-cyan-950/40 text-cyan-400 border border-cyan-500/30';
      case 'queued':
        return 'bg-amber-950/40 text-amber-400 border border-amber-500/30';
      case 'failed':
        return 'bg-rose-950/40 text-rose-400 border border-rose-500/30';
      default:
        return 'bg-slate-800 text-slate-400 border border-slate-700';
    }
  };

  const getSourceTypeBadgeClass = (type: string) => {
    switch (type.toLowerCase()) {
      case 'pdf':
        return 'bg-red-950/30 text-red-400 border border-red-500/20';
      case 'docx':
        return 'bg-blue-950/30 text-blue-400 border border-blue-500/20';
      case 'png':
      case 'jpg':
      case 'jpeg':
        return 'bg-purple-950/30 text-purple-400 border border-purple-500/20';
      default:
        return 'bg-slate-850 text-slate-400 border border-slate-700/50';
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col antialiased">
      {/* Background Glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="sticky top-0 z-40 bg-slate-900/60 backdrop-blur-md border-b border-slate-800/80 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-3.5 h-3.5 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.6)] animate-pulse" />
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-slate-50 to-cyan-400 bg-clip-text text-transparent">
            IDIP Document Portal
          </h1>
        </div>
        <div className="flex items-center gap-4">
          {/* Health status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/40 text-sm">
            <span className={`w-2 h-2 rounded-full ${
              systemHealth === 'healthy' ? 'bg-emerald-500' :
              systemHealth === 'degraded' ? 'bg-amber-500' : 'bg-rose-500'
            }`} />
            <span className="text-slate-300 font-medium">
              System: {systemHealth.toUpperCase()}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 flex flex-col gap-6">
        
        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800/80 pb-px">
          <button
            onClick={() => setActiveTab('documents')}
            className={`px-5 py-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all duration-200 ${
              activeTab === 'documents'
                ? 'border-cyan-400 text-cyan-400 bg-cyan-950/10'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <FileText className="w-4 h-4" />
            Documents Catalogue
          </button>
          <button
            onClick={() => {
              setActiveTab('chat');
              fetchChatDocuments();
            }}
            className={`px-5 py-3 text-xs font-semibold flex items-center gap-2 border-b-2 transition-all duration-200 ${
              activeTab === 'chat'
                ? 'border-cyan-400 text-cyan-400 bg-cyan-950/10'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            AI Chat Assistant / Q&A
          </button>
        </div>

        {activeTab === 'documents' && (
          <>
            {/* Top Section: Upload & Live Pipeline Status */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Drag & Drop Card */}
              <div className="lg:col-span-2 bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-6 flex flex-col gap-4">
                <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-cyan-400" />
                  Upload Document to Ingestion Pipeline
                </h2>

                <div
                  {...getRootProps()}
                  className={`flex-1 border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-300 ${
                    isDragActive
                      ? 'border-cyan-500 bg-cyan-950/20'
                      : 'border-slate-800 hover:border-slate-700 bg-slate-950/20'
                  }`}
                >
                  <input {...getInputProps()} />
                  <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400 group-hover:text-slate-300 transition-colors">
                    <Upload className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-slate-200 font-medium">
                      {isDragActive ? "Drop the file here" : "Drag & drop document here, or click to browse"}
                    </p>
                    <p className="text-xs text-slate-500 mt-1.5">
                      PDF, DOCX, PNG, JPG up to 50MB
                    </p>
                  </div>
                </div>

                {/* Active upload status */}
                {uploadingFile && (
                  <div className="bg-slate-950/65 border border-slate-800/80 rounded-xl p-4 flex flex-col gap-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-slate-300 font-medium flex items-center gap-2">
                        <FileText className="w-3.5 h-3.5 text-cyan-400" />
                        {uploadingFile.name}
                      </span>
                      <span className="text-cyan-400 font-semibold">{uploadProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-cyan-400 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {uploadError && (
                  <div className="bg-rose-950/30 border border-rose-500/20 rounded-xl p-4 flex items-center gap-3 text-sm text-rose-400">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{uploadError}</span>
                  </div>
                )}
              </div>

              {/* SSE Live Pipeline Status */}
              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-6 flex flex-col gap-4">
                <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-400" />
                  Live Pipeline Tracker
                </h2>

                {activePipeline ? (
                  <div className="flex-1 flex flex-col gap-4">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-200 truncate max-w-[200px]">
                          {activePipeline.filename}
                        </h3>
                        <p className="text-[11px] text-slate-500 font-mono mt-0.5">
                          ID: {activePipeline.docId.slice(0, 8)}...
                        </p>
                      </div>
                      <span className="text-[10px] bg-purple-950/30 text-purple-400 border border-purple-500/20 px-2 py-0.5 rounded-full font-medium">
                        Tracking
                      </span>
                    </div>

                    <div className="relative flex-1 flex flex-col gap-3 pl-6 border-l border-slate-800 mt-2 ml-3">
                      {activePipeline.steps.map((step, idx) => (
                        <div key={idx} className="relative flex items-start justify-between">
                          {/* Step Indicator Dot */}
                          <div className="absolute -left-[31px] top-1">
                            {step.status === 'completed' && (
                              <div className="w-4 h-4 rounded-full bg-emerald-500 border-4 border-slate-950 flex items-center justify-center text-[8px] text-slate-950">
                                <Check className="w-2.5 h-2.5 text-slate-950 stroke-[3]" />
                              </div>
                            )}
                            {step.status === 'processing' && (
                              <div className="w-4 h-4 rounded-full bg-slate-950 border-2 border-cyan-400 flex items-center justify-center">
                                <Loader2 className="w-2.5 h-2.5 text-cyan-400 animate-spin" />
                              </div>
                            )}
                            {step.status === 'failed' && (
                              <div className="w-4 h-4 rounded-full bg-rose-500 border-4 border-slate-950 flex items-center justify-center">
                                <AlertCircle className="w-2.5 h-2.5 text-slate-950 stroke-[3]" />
                              </div>
                            )}
                            {step.status === 'pending' && (
                              <div className="w-3.5 h-3.5 rounded-full bg-slate-800 border-2 border-slate-950" />
                            )}
                          </div>

                          <div className="flex flex-col gap-0.5">
                            <span className={`text-xs font-semibold ${
                              step.status === 'completed' ? 'text-slate-200' :
                              step.status === 'processing' ? 'text-cyan-400' :
                              step.status === 'failed' ? 'text-rose-400' : 'text-slate-500'
                            }`}>
                              {step.stepName}
                            </span>
                            {step.error && (
                              <span className="text-[10px] text-rose-500 font-medium">
                                {step.error}
                              </span>
                            )}
                          </div>

                          {step.status !== 'pending' && (
                            <span className="text-[10px] text-slate-500 font-mono mt-0.5">
                              {step.elapsed}s
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-slate-850 rounded-xl bg-slate-950/20">
                    <Clock className="w-8 h-8 text-slate-600 mb-2 stroke-[1.5]" />
                    <p className="text-xs text-slate-400 max-w-[200px]">
                      No active ingestion to track. Upload a document to view progression in real time.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Filters and Search Bar */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-6 flex flex-col gap-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <Filter className="w-4 h-4 text-cyan-400" />
                  Document Catalogue & Filters
                </h2>
                <button
                  onClick={() => fetchDocuments()}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-750 text-xs font-semibold flex items-center gap-1.5 transition-colors border border-slate-700/40"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Refresh
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                {/* Status Filter */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-medium">Pipeline Status</label>
                  <select
                    value={statusFilter}
                    onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  >
                    <option value="">All Statuses</option>
                    <option value="completed">Completed</option>
                    <option value="processing">Processing</option>
                    <option value="queued">Queued</option>
                    <option value="failed">Failed</option>
                  </select>
                </div>

                {/* Source Type Filter */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-medium">Source Type</label>
                  <select
                    value={sourceTypeFilter}
                    onChange={(e) => { setSourceTypeFilter(e.target.value); setPage(1); }}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  >
                    <option value="">All Source Types</option>
                    <option value="pdf">PDF</option>
                    <option value="docx">DOCX</option>
                    <option value="png">PNG</option>
                    <option value="jpg">JPG / JPEG</option>
                  </select>
                </div>

                {/* Start Date */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-medium flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    Start Date
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => { setStartDate(e.target.value); setPage(1); }}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  />
                </div>

                {/* End Date */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-slate-400 font-medium flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    End Date
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => { setEndDate(e.target.value); setPage(1); }}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors"
                  />
                </div>
              </div>

              {/* Bulk Action Bar */}
              {selectedDocIds.size > 0 && (
                <div className="flex items-center justify-between bg-cyan-950/20 border border-cyan-500/30 rounded-xl px-4 py-3 animate-fadeIn">
                  <span className="text-xs text-cyan-300 font-medium">
                    {selectedDocIds.size} {selectedDocIds.size === 1 ? 'document' : 'documents'} selected
                  </span>
                  <button
                    onClick={() => setShowBulkDeleteConfirm(true)}
                    className="bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 border border-rose-500/30 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Bulk Delete
                  </button>
                </div>
              )}

              {/* Document Table */}
              <div className="overflow-x-auto rounded-xl border border-slate-850">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-950/60 border-b border-slate-850 text-slate-400 font-medium">
                      <th className="py-3.5 px-4 w-10">
                        <input
                          type="checkbox"
                          checked={documents.length > 0 && selectedDocIds.size === documents.length}
                          onChange={toggleSelectAll}
                          className="rounded border-slate-700 text-cyan-500 bg-slate-950 focus:ring-cyan-500/30 w-3.5 h-3.5"
                        />
                      </th>
                      <th className="py-3.5 px-4 font-semibold">Filename</th>
                      <th className="py-3.5 px-4 font-semibold w-28">Source Type</th>
                      <th className="py-3.5 px-4 font-semibold w-32">Status</th>
                      <th className="py-3.5 px-4 font-semibold w-32">Doc Type</th>
                      <th className="py-3.5 px-4 font-semibold w-36">Ingested</th>
                      <th className="py-3.5 px-4 font-semibold text-center w-40">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-850/60 bg-slate-900/20">
                    {documents.length > 0 ? (
                      documents.map((doc) => (
                        <tr
                          key={doc.doc_id}
                          className={`hover:bg-slate-850/40 transition-colors ${
                            selectedDocIds.has(doc.doc_id) ? 'bg-cyan-950/5' : ''
                          }`}
                        >
                          <td className="py-3.5 px-4">
                            <input
                              type="checkbox"
                              checked={selectedDocIds.has(doc.doc_id)}
                              onChange={() => toggleSelectDoc(doc.doc_id)}
                              className="rounded border-slate-700 text-cyan-500 bg-slate-950 focus:ring-cyan-500/30 w-3.5 h-3.5"
                            />
                          </td>
                          <td className="py-3.5 px-4 font-medium text-slate-200 truncate max-w-[220px]">
                            {doc.filename}
                          </td>
                          <td className="py-3.5 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${getSourceTypeBadgeClass(doc.source_type)}`}>
                              {doc.source_type}
                            </span>
                          </td>
                          <td className="py-3.5 px-4">
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide ${getStatusBadgeClass(doc.status)}`}>
                              {doc.status}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-slate-300 font-medium">
                            {doc.doc_type ? (
                              <span className="flex items-center gap-1.5 text-slate-300">
                                <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                                {doc.doc_type.charAt(0).toUpperCase() + doc.doc_type.slice(1)}
                              </span>
                            ) : (
                              <span className="text-slate-600 font-normal italic">Unclassified</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-slate-400 font-medium flex items-center gap-1.5 mt-1.5">
                            <Clock className="w-3.5 h-3.5 text-slate-500" />
                            {formatRelativeTime(doc.ingestion_ts)}
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="flex items-center justify-center gap-2">
                              {doc.status.toLowerCase() === 'completed' && (
                                <button
                                  onClick={() => {
                                    setChatScopeDocId(doc.doc_id);
                                    setActiveTab('chat');
                                    fetchChatDocuments();
                                  }}
                                  className="p-1.5 rounded bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 transition-colors"
                                  title="Ask AI about this document"
                                >
                                  <MessageSquare className="w-3.5 h-3.5" />
                                </button>
                              )}
                              <button
                                onClick={() => handleViewDetails(doc.doc_id)}
                                className="p-1.5 rounded bg-slate-800/80 hover:bg-slate-750 text-slate-300 transition-colors border border-slate-700/30"
                                title="View Metadata"
                              >
                                <Eye className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleReprocess(doc.doc_id, doc.filename)}
                                className="p-1.5 rounded bg-slate-800/80 hover:bg-slate-750 text-slate-300 transition-colors border border-slate-700/30"
                                title="Re-process Document"
                              >
                                <RefreshCw className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => setDeletingDocId(doc.doc_id)}
                                className="p-1.5 rounded bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 transition-colors"
                                title="Delete"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-slate-500 bg-slate-900/10">
                          <FileText className="w-10 h-10 mx-auto mb-2 text-slate-600 stroke-[1.5]" />
                          <p className="font-semibold text-slate-400 text-sm">No documents found</p>
                          <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                            There are no documents matching the selected filters. Upload new documents using the dropzone at the top.
                          </p>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination controls */}
              {total > limit && (
                <div className="flex items-center justify-between border-t border-slate-850 pt-4">
                  <span className="text-xs text-slate-400">
                    Showing <strong className="text-slate-200">{Math.min(total, (page - 1) * limit + 1)}</strong> to{' '}
                    <strong className="text-slate-200">{Math.min(total, page * limit)}</strong> of{' '}
                    <strong className="text-slate-200">{total}</strong> entries
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      disabled={page === 1}
                      onClick={() => setPage(p => Math.max(1, p - 1))}
                      className="p-1.5 rounded bg-slate-850 hover:bg-slate-800 border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-slate-300"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-xs font-semibold text-slate-200 px-3 py-1 bg-slate-900 rounded-lg border border-slate-800">
                      Page {page} of {Math.ceil(total / limit)}
                    </span>
                    <button
                      disabled={page >= Math.ceil(total / limit)}
                      onClick={() => setPage(p => p + 1)}
                      className="p-1.5 rounded bg-slate-850 hover:bg-slate-800 border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-slate-300"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {activeTab === 'chat' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-[500px]">
            {/* Left Sidebar: Controls & Suggestions */}
            <div className="lg:col-span-1 flex flex-col gap-6">
              
              {/* Document Scope Card */}
              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Filter className="w-3.5 h-3.5 text-cyan-400" />
                  Document Scope
                </h3>
                <p className="text-[11px] text-slate-500 font-medium leading-relaxed">
                  Select which document context the AI assistant should search for answers:
                </p>
                <select
                  value={chatScopeDocId}
                  onChange={(e) => setChatScopeDocId(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors w-full cursor-pointer"
                >
                  <option value="">🔍 Search All Documents</option>
                  {chatDocuments.map(doc => (
                    <option key={doc.doc_id} value={doc.doc_id}>
                      📄 {doc.filename.length > 25 ? `${doc.filename.slice(0, 22)}...` : doc.filename}
                    </option>
                  ))}
                </select>
              </div>

              {/* Suggested Questions Card */}
              <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-5 flex flex-col gap-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                  Suggested Questions
                </h3>
                <div className="flex flex-col gap-2">
                  <button
                    disabled={isGenerating}
                    onClick={() => handleSendQuery("What is the invoice total?")}
                    className="text-left w-full px-3 py-2.5 bg-slate-950/45 hover:bg-slate-800/50 border border-slate-800/85 hover:border-slate-700 text-[11px] font-medium text-slate-300 rounded-xl transition-all duration-200"
                  >
                    🔍 "What is the invoice total?"
                  </button>
                  <button
                    disabled={isGenerating}
                    onClick={() => handleSendQuery("Who signed this contract?")}
                    className="text-left w-full px-3 py-2.5 bg-slate-950/45 hover:bg-slate-800/50 border border-slate-800/85 hover:border-slate-700 text-[11px] font-medium text-slate-300 rounded-xl transition-all duration-200"
                  >
                    📄 "Who signed the contract?"
                  </button>
                  <button
                    disabled={isGenerating}
                    onClick={() => handleSendQuery("Summarize document performance")}
                    className="text-left w-full px-3 py-2.5 bg-slate-950/45 hover:bg-slate-800/50 border border-slate-800/85 hover:border-slate-700 text-[11px] font-medium text-slate-300 rounded-xl transition-all duration-200"
                  >
                    📈 "Summarize document performance"
                  </button>
                </div>
              </div>
            </div>

            {/* Right Column: Active Chat Area */}
            <div className="lg:col-span-3 bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl flex flex-col h-[600px] overflow-hidden">
              
              {/* Chat Header */}
              <div className="px-5 py-4 border-b border-slate-800 bg-slate-900/40 flex justify-between items-center">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-cyan-950/40 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <Bot className="w-4.5 h-4.5 text-cyan-400" />
                  </div>
                  <div>
                    <h3 className="text-xs font-semibold text-slate-200">
                      IDIP RAG Chat Assistant
                    </h3>
                    <p className="text-[10px] text-slate-500 font-medium">
                      Answering directly from your indexed documents
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setChatHistory([{
                    id: 'welcome',
                    sender: 'assistant',
                    text: "Hello! I am your IDIP AI Assistant. Ask me questions about your uploaded documents, and I'll retrieve the relevant chunks to answer you, complete with citation tags and confidence metrics!"
                  }])}
                  className="px-2.5 py-1.5 text-[10px] font-semibold bg-slate-850 hover:bg-slate-800 border border-slate-750/40 text-slate-300 rounded-lg flex items-center gap-1 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear Chat
                </button>
              </div>

              {/* Chat Message Box */}
              <div className="flex-1 p-5 overflow-y-auto flex flex-col gap-4">
                {chatHistory.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-3 max-w-[85%] ${
                      msg.sender === 'user' ? 'self-end flex-row-reverse' : 'self-start'
                    }`}
                  >
                    {/* Icon indicator */}
                    <div className={`w-8 h-8 rounded-full shrink-0 flex items-center justify-center border text-[10px] ${
                      msg.sender === 'user'
                        ? 'bg-cyan-950/20 border-cyan-500/30 text-cyan-400'
                        : 'bg-purple-950/20 border-purple-500/30 text-purple-400'
                    }`}>
                      {msg.sender === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
                    </div>

                    {/* Speech bubbles */}
                    <div className="flex flex-col gap-2">
                      <div className={`rounded-2xl px-4 py-3 border text-xs font-medium leading-relaxed ${
                        msg.sender === 'user'
                          ? 'bg-slate-900 border-cyan-500/25 text-slate-200 rounded-tr-none'
                          : 'bg-slate-950 border-purple-500/20 text-slate-300 rounded-tl-none shadow-sm shadow-purple-500/5'
                      }`}>
                        <p className="whitespace-pre-wrap">
                          {msg.sender === 'assistant' && msg.text.startsWith('Answer:') 
                            ? msg.text.substring(7).trim() 
                            : msg.text}
                        </p>
                        
                        {/* Typing cursor animation when streaming */}
                        {msg.streaming && (
                          <span className="inline-block w-1.5 h-3.5 ml-1 bg-cyan-400 animate-pulse align-middle" />
                        )}
                      </div>

                      {/* AI metadata block (Citations, Confidence) */}
                      {msg.sender === 'assistant' && !msg.streaming && (msg.confidence !== undefined || (msg.citations && msg.citations.length > 0)) && (
                        <div className="flex flex-col gap-2 pl-1 animate-fadeIn">
                          
                          {/* Confidence score indicator */}
                          {msg.confidence !== undefined && (
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] text-slate-500 font-medium">Confidence Score:</span>
                              <div className="flex items-center gap-1.5">
                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                                  msg.confidence >= 0.85
                                    ? 'bg-emerald-950/40 text-emerald-400 border border-emerald-500/20'
                                    : msg.confidence >= 0.70
                                    ? 'bg-amber-950/40 text-amber-400 border border-amber-500/20'
                                    : 'bg-rose-950/40 text-rose-400 border border-rose-500/20'
                                }`}>
                                  {Math.round(msg.confidence * 100)}%
                                </span>
                                <div className="w-20 bg-slate-800 rounded-full h-1 overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${
                                      msg.confidence >= 0.85 ? 'bg-emerald-500' : msg.confidence >= 0.70 ? 'bg-amber-500' : 'bg-rose-500'
                                    }`}
                                    style={{ width: `${msg.confidence * 100}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Low confidence warning banner */}
                          {msg.lowConfidence && (
                            <div className="bg-rose-950/20 border border-rose-500/20 rounded-xl p-2.5 flex items-start gap-2 max-w-md mt-1 text-[10px] text-rose-400 font-medium">
                              <ShieldAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                              <span>
                                Warning: The response has low model confidence. Please verify details with source files.
                              </span>
                            </div>
                          )}

                          {/* Citations section */}
                          {msg.citations && msg.citations.length > 0 && (
                            <div className="flex flex-col gap-1.5 mt-1">
                              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                                Citations & Sources ({msg.citations.length})
                              </span>
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl">
                                {msg.citations.map((cite, idx) => (
                                  <div
                                    key={idx}
                                    className="bg-slate-950/75 border border-slate-850 p-2.5 rounded-xl flex flex-col gap-1.5 hover:border-slate-700 transition-colors"
                                  >
                                    <div className="flex justify-between items-center text-[9px] font-semibold">
                                      <span className="bg-purple-950/40 text-purple-400 border border-purple-500/20 px-1.5 py-0.5 rounded-full uppercase">
                                        [DOC-{cite.doc_id_short}]
                                      </span>
                                      <button
                                        onClick={() => handleViewDetails(cite.doc_id)}
                                        className="text-cyan-400 hover:text-cyan-300 flex items-center gap-0.5 transition-colors"
                                        title="View document metadata"
                                      >
                                        Details
                                        <ExternalLink className="w-2.5 h-2.5" />
                                      </button>
                                    </div>
                                    <p className="text-[10px] text-slate-300 line-clamp-2 select-text font-medium leading-relaxed italic">
                                      "{cite.text_snippet}"
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Copy answer action button */}
                          <div className="flex items-center gap-2 mt-1 border-t border-slate-850/30 pt-1.5">
                            <button
                              onClick={() => handleCopyAnswer(msg.text)}
                              className="text-[10px] text-slate-400 hover:text-slate-200 flex items-center gap-1.5 transition-colors"
                            >
                              <Copy className="w-3 h-3" />
                              Copy Answer
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {/* Scroll Anchor */}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input form */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendQuery();
                }}
                className="p-4 border-t border-slate-800/80 bg-slate-950/30 flex gap-2.5 items-center"
              >
                <div className="relative flex-1">
                  <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <input
                    type="text"
                    value={chatQuery}
                    onChange={(e) => setChatQuery(e.target.value)}
                    disabled={isGenerating}
                    placeholder={
                      isGenerating
                        ? "Streaming AI generation..."
                        : chatScopeDocId
                        ? "Ask about selected document context..."
                        : "Ask about any contract, invoice, or report..."
                    }
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-3 text-xs text-slate-200 focus:outline-none focus:border-cyan-500/50 transition-colors placeholder-slate-600 disabled:opacity-50"
                  />
                </div>
                <button
                  type="submit"
                  disabled={!chatQuery.trim() || isGenerating}
                  className="bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 text-slate-950 disabled:text-slate-600 px-4 py-3 rounded-xl text-xs font-bold transition-all duration-200 flex items-center gap-1.5 shadow-[0_0_12px_rgba(6,182,212,0.15)] hover:shadow-[0_0_16px_rgba(6,182,212,0.3)] disabled:shadow-none shrink-0"
                >
                  {isGenerating ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5 stroke-[2.5]" />
                      Send
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        )}
      </main>

      {/* VIEW MODAL */}
      {viewingDocDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl animate-scaleIn">
            <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/60">
              <h3 className="font-bold text-slate-100 flex items-center gap-2 text-sm">
                <Info className="w-4.5 h-4.5 text-cyan-400" />
                Document Metadata & Extraction Details
              </h3>
              <button
                onClick={() => setViewingDocDetails(null)}
                className="text-slate-400 hover:text-slate-200 transition-colors text-sm font-semibold"
              >
                Close
              </button>
            </div>
            <div className="p-6 max-h-[60vh] overflow-y-auto flex flex-col gap-4 text-xs font-medium">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">Document Identifier</span>
                  <span className="text-slate-200 font-mono select-all truncate">{viewingDocDetails.doc_id}</span>
                </div>
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">Source URI</span>
                  <span className="text-slate-200 select-all truncate">{viewingDocDetails.source_uri}</span>
                </div>
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">Size (Bytes)</span>
                  <span className="text-slate-200">{formatBytes(viewingDocDetails.byte_size)}</span>
                </div>
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">Language Detected</span>
                  <span className="text-slate-200 uppercase font-semibold">{viewingDocDetails.language}</span>
                </div>
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">MIME Type</span>
                  <span className="text-slate-200">{viewingDocDetails.mime_type}</span>
                </div>
                <div className="bg-slate-950/60 border border-slate-850 p-3.5 rounded-xl flex flex-col gap-1">
                  <span className="text-[10px] text-slate-500">Checksum (SHA-256)</span>
                  <span className="text-slate-200 font-mono truncate select-all">{viewingDocDetails.checksum}</span>
                </div>
              </div>

              {/* Extracted Entities */}
              <div className="bg-slate-950/60 border border-slate-850 p-4 rounded-xl flex flex-col gap-3">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Extracted NER Entities</span>
                {viewingDocDetails.extracted_entities && viewingDocDetails.extracted_entities.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {viewingDocDetails.extracted_entities.map((ent: any, idx: number) => (
                      <span
                        key={idx}
                        className="bg-purple-950/30 text-purple-300 border border-purple-500/25 px-2.5 py-0.5 rounded text-[10px] font-semibold"
                      >
                        {ent.text} ({ent.label})
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-600 font-normal italic text-xs">No named entities extracted yet.</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SINGLE DELETE CONFIRM MODAL */}
      {deletingDocId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-850 rounded-2xl w-full max-w-sm overflow-hidden shadow-2xl p-6 flex flex-col gap-4 animate-scaleIn">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-rose-500" />
              Confirm Deletion
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed font-medium">
              Are you sure you want to permanently delete this document? This action deletes database records, features, and vector store indices.
            </p>
            <div className="flex justify-end gap-2.5 text-xs font-semibold">
              <button
                onClick={() => setDeletingDocId(null)}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 transition-colors border border-slate-700/30"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteConfirm}
                className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-slate-50 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* BULK DELETE CONFIRM MODAL */}
      {showBulkDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-850 rounded-2xl w-full max-w-sm overflow-hidden shadow-2xl p-6 flex flex-col gap-4 animate-scaleIn">
            <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-rose-500" />
              Confirm Bulk Deletion
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed font-medium">
              Are you sure you want to delete all <strong className="text-rose-400">{selectedDocIds.size}</strong> selected documents? This action is permanent and cannot be undone.
            </p>
            <div className="flex justify-end gap-2.5 text-xs font-semibold">
              <button
                onClick={() => setShowBulkDeleteConfirm(false)}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-750 text-slate-300 transition-colors border border-slate-700/30"
              >
                Cancel
              </button>
              <button
                onClick={handleBulkDeleteConfirm}
                className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-slate-50 transition-colors"
              >
                Delete All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
