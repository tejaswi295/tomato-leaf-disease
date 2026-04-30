import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Camera, Image as ImageIcon, ScanLine, RefreshCw, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import Alert from '../components/Alert';

export default function Classification() {
  const [source, setSource] = useState('upload');
  const [imageSrc, setImageSrc] = useState(null);
  const [file, setFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [error, setError] = useState(null);

  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImageSrc(imageSrc);
    fetch(imageSrc)
      .then(res => res.blob())
      .then(blob => {
        const f = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        setFile(f);
      });
  }, [webcamRef]);

  const handleFileUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setImageSrc(URL.createObjectURL(selectedFile));
    }
  };

  const handleClassify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setGradcam(null);

    const formData = new FormData();
    formData.append('file', file);
    const API_BASE = import.meta.env.VITE_API_URL || 'https://tomato-leaf-disease-1-sf8o.onrender.com';


    try {
      const predResponse = await axios.post(`${API_BASE}/predict/`, formData);
      setResult(predResponse.data);

      const gradcamResponse = await axios.post(`${API_BASE}/gradcam/`, formData);
      setGradcam(gradcamResponse.data.gradcam_base64);

    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setImageSrc(null);
    setFile(null);
    setResult(null);
    setGradcam(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const chartData = result && result.class_probabilities ? Object.keys(result.class_probabilities).map(key => ({
    name: key.replace('Tomato___', '').replace(/_/g, ' '),
    probability: result.class_probabilities[key]
  })).sort((a, b) => b.probability - a.probability).slice(0, 5) : [];

  return (
    <div className="space-y-6 animate-page-in">
      {/* Header Block */}
      <div>
        <h2 className="text-3xl font-semibold text-app-text mb-2">
          Diagnostic Scanner
        </h2>
        <p className="text-app-textSecondary">Upload or capture a tomato leaf image to detect diseases.</p>
      </div>

      {/* Input Toggle */}
      <div className="flex gap-3">
        <button
          onClick={() => { setSource('upload'); resetState(); }}
          className={`px-5 py-2 text-sm font-medium rounded-xl transition-all flex items-center gap-2 border cursor-pointer active:scale-95 ${source === 'upload' ? 'bg-app-primaryLight text-app-primary border-blue-200' : 'bg-white text-app-textSecondary hover:text-app-text border-app-border hover:border-slate-300'}`}
        >
          <ImageIcon className="w-4 h-4" /> Local File
        </button>
        <button
          onClick={() => { setSource('camera'); resetState(); }}
          className={`px-5 py-2 text-sm font-medium rounded-xl transition-all flex items-center gap-2 border cursor-pointer active:scale-95 ${source === 'camera' ? 'bg-app-primaryLight text-app-primary border-blue-200' : 'bg-white text-app-textSecondary hover:text-app-text border-app-border hover:border-slate-300'}`}
        >
          <Camera className="w-4 h-4" /> Live Camera
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

        {/* Input Card */}
        <div className="bg-white border border-app-border rounded-2xl flex flex-col pt-6 pb-6 px-6 shadow-sm">
          <div className="text-sm font-medium text-app-textSecondary mb-4 flex justify-between items-center">
            <span>Input Image</span>
            {file && <span className="text-app-primary flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4" /> Loaded</span>}
          </div>

          <div className="flex-1 min-h-[350px] relative bg-slate-50 rounded-xl border border-dashed border-slate-300 flex items-center justify-center overflow-hidden transition-all group">
            {imageSrc ? (
              <img src={imageSrc} alt="Input" className="absolute inset-0 w-full h-full object-contain" />
            ) : source === 'camera' ? (
              <div className="text-center w-full h-full relative">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ facingMode: "environment" }}
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <button onClick={capture} className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-app-primary text-white px-6 py-2.5 rounded-full font-medium shadow-lg shadow-blue-200 hover:bg-app-primaryHover flex items-center gap-2 transition-transform active:scale-95 cursor-pointer">
                  <Camera className="w-5 h-5" /> Capture Frame
                </button>
              </div>
            ) : (
              <div className="text-center w-full">
                <div className="text-slate-300 mb-4 group-hover:text-app-primary transition-colors">
                  <ImageIcon className="w-12 h-12 mx-auto" />
                </div>
                <p className="text-lg text-app-text">Drag and drop leaf image</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  className="hidden"
                  accept="image/jpeg, image/png"
                />
                <button onClick={() => fileInputRef.current.click()} className="mt-4 border border-app-border text-app-textSecondary hover:text-app-text hover:border-slate-400 rounded-lg px-6 py-2 font-medium transition-all cursor-pointer active:scale-95">
                  Browse Files
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 flex flex-col">
              <Alert type="error" message={error} />
            </div>
          )}

          <div className="mt-6 flex gap-3">
            <button
              id="classify-btn"
              onClick={handleClassify}
              disabled={!file || loading}
              className={`flex-1 py-3 rounded-xl font-medium text-lg flex justify-center items-center gap-2 transition-all cursor-pointer ${!file || loading ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' : 'bg-app-primary text-white hover:bg-app-primaryHover shadow-lg shadow-blue-200 active:scale-95'}`}
            >
              {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <><ScanLine className="w-5 h-5" /> Run Diagnostics</>}
            </button>
            {imageSrc && (
              <button onClick={resetState} className="px-6 rounded-xl border border-app-border hover:bg-slate-50 text-app-text font-medium transition-colors cursor-pointer active:scale-95">
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Output Card */}
        <div className="bg-white border border-app-border rounded-2xl flex flex-col pt-6 pb-6 px-6 shadow-sm">
          <div className="text-sm font-medium text-app-textSecondary mb-4 flex justify-between items-center">
            <span>Analysis Results</span>
            {result && <span className="text-app-primary flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4" /> Complete</span>}
          </div>

          {result ? (
            <div className="flex-1 flex flex-col animate-page-in">

              {/* Result Title */}
              <div className="bg-slate-50 border border-app-border rounded-xl p-5 mb-6">
                <div className="mb-2 text-sm text-app-textSecondary font-medium">Identified Condition</div>
                <div className="text-3xl md:text-4xl font-bold text-app-text tracking-tight">
                  {result.predicted_class.replace('Tomato___', '').replace(/_/g, ' ')}
                </div>
                <div className="mt-3 flex items-center gap-3">
                  <span className="text-sm text-app-textSecondary">Confidence Score</span>
                  <div className={`px-2.5 py-1 rounded-md border font-semibold ${result.confidence_score < 0.7 ? 'bg-amber-50 text-amber-700 border-amber-200' : 'bg-app-primaryLight border-blue-200 text-app-primary'}`}>
                    {(result.confidence_score * 100).toFixed(1)}%
                  </div>
                  {result.confidence_score < 0.7 && (
                    <div className="flex items-center gap-1.5 ml-2 text-amber-600 text-sm font-medium">
                      <AlertTriangle className="w-4 h-4" />
                      <span>Low confidence — try a clearer image</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-2 flex-1">
                {/* Probabilities */}
                <div className="flex flex-col bg-slate-50 rounded-xl p-4 border border-app-border">
                  <h4 className="text-sm font-medium text-app-text mb-4">Top Matches</h4>
                  <div className="flex-1 min-h-[180px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
                        <XAxis type="number" domain={[0, 1]} hide />
                        <YAxis dataKey="name" type="category" width={110} axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#475569' }} />
                        <Tooltip
                          cursor={{ fill: '#f1f5f9' }}
                          wrapperStyle={{ zIndex: 100 }}
                          contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px' }}
                          itemStyle={{ color: '#0f172a', fontWeight: 600 }}
                          formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Confidence']}
                        />
                        <Bar dataKey="probability" barSize={10} radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={index === 0 ? '#2563eb' : '#cbd5e1'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Heatmap */}
                <div className="flex flex-col bg-slate-50 rounded-xl p-4 border border-app-border">
                  <h4 className="text-sm font-medium text-app-text mb-4">Feature Attention (Grad-CAM)</h4>
                  <div className="flex-1 bg-white rounded-lg relative flex justify-center items-center overflow-hidden min-h-[180px] border border-app-border">
                    {gradcam ? (
                      <>
                        <img src={`data:image/jpeg;base64,${gradcam}`} alt="Grad-CAM" className="absolute inset-0 w-full h-full object-cover" />
                      </>
                    ) : (
                      <div className="flex flex-col items-center">
                        <RefreshCw className="w-6 h-6 animate-spin text-app-textSecondary mb-3" />
                        <span className="text-app-textSecondary text-sm">Rendering heatmap...</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

            </div>
          ) : (
            <div className="flex-1 border border-dashed border-slate-300 rounded-xl bg-slate-50 flex flex-col justify-center items-center text-slate-400">
              <ScanLine className="w-12 h-12 mb-3 opacity-50" />
              <p className="font-medium text-app-textSecondary">Awaiting scan execution</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
