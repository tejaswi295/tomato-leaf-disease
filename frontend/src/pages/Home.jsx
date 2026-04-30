import React, { useState, useRef } from 'react';
import axios from 'axios';
import { 
  Activity, 
  RefreshCw, 
  Image as ImageIcon, 
  Server,
  LayoutTemplate,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import Alert from '../components/Alert';

export default function Home({ setActiveTab }) {
  // Gradcam states
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);

  const API_BASE = import.meta.env.VITE_API_URL || 'https://tomato-leaf-disease-2-sf8o.onrender.com';

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setError(null);
      setGradcamImage(null);
    }
  };

  const runGradCam = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await axios.post(`${API_BASE}/gradcam/`, formData);
      setGradcamImage(`data:image/jpeg;base64,${res.data.gradcam_base64}`);
    } catch (err) {
      setError("Grad-CAM generation failed. " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Mock table data
  const recentDiagnostics = [
    { id: 'SCN-8092', disease: 'Tomato Early blight', confidence: '99.2%', date: '2 Mins Ago', status: 'Warning' },
    { id: 'SCN-8091', disease: 'Tomato Healthy', confidence: '98.8%', date: '15 Mins Ago', status: 'Clear' },
    { id: 'SCN-8090', disease: 'Tomato Leaf Mold', confidence: '94.5%', date: '1 Hour Ago', status: 'Warning' },
    { id: 'SCN-8089', disease: 'Tomato Target Spot', confidence: '89.1%', date: '3 Hours Ago', status: 'Warning' },
  ];

  return (
    <div className="animate-page-in">
      {/* Compact Dashboard Header */}
      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-app-text mb-2">Dashboard</h1>
      </header>

      {/* Grad-CAM Workspace */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-app-text mb-4">Diagnostic Tooling</h2>
        <div className="bg-white border border-app-border rounded-2xl p-6 md:p-10 shadow-sm relative">
           <h3 className="text-lg font-medium text-app-text mb-6 flex items-center gap-2">
             <Activity className="w-5 h-5 text-app-primary" /> 
             Grad-CAM Heatmap Scanner
           </h3>

           {error && <div className="mb-6"><Alert type="error" message={error} /></div>}

           <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-2">
             {/* Upload Zone */}
             <div>
                <div className="mb-2 text-sm font-medium text-app-textSecondary">Input Suspect Image</div>
                <div 
                  onClick={() => fileInputRef.current.click()} 
                  className="border-2 border-dashed border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100 transition-colors rounded-xl h-[300px] flex flex-col items-center justify-center cursor-pointer group relative overflow-hidden"
                >
                   {previewUrl ? (
                     <img src={previewUrl} alt="Preview" className="absolute inset-0 w-full h-full object-cover" />
                   ) : (
                     <div className="text-center px-4">
                       <ImageIcon className="w-12 h-12 text-slate-300 mb-4 mx-auto group-hover:text-app-primary transition-colors" />
                       <p className="text-app-text font-medium">Click to upload leaf image</p>
                       <p className="text-app-textSecondary text-sm mt-1">JPEG or PNG</p>
                     </div>
                   )}
                </div>
                <input 
                   type="file" 
                   ref={fileInputRef} 
                   onChange={handleFileUpload} 
                   className="hidden" 
                   accept="image/jpeg, image/png" 
                />
                
                <div className="mt-4">
                    <button 
                      onClick={runGradCam} 
                      disabled={!file || loading}
                      className={`w-full py-3.5 rounded-xl font-medium flex justify-center items-center gap-2 cursor-pointer transition-all ${!file || loading ? 'bg-slate-100 text-slate-400 border border-slate-200' : 'bg-app-primary text-white hover:bg-app-primaryHover shadow-lg shadow-app-primaryLight active:scale-[0.98]'}`}
                    >
                      {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : "Deploy Heatmap Generation"}
                    </button>
                </div>
             </div>

             {/* Result Zone */}
             <div>
                <div className="mb-2 text-sm font-medium text-app-textSecondary">Visualization Output</div>
                <div className="border border-app-border bg-slate-50 rounded-xl h-[300px] flex flex-col items-center justify-center p-6 relative overflow-hidden">
                   {loading ? (
                      <div className="text-center">
                        <RefreshCw className="w-8 h-8 text-app-textSecondary animate-spin mb-4 mx-auto" />
                        <p className="text-app-textSecondary font-medium">Processing spatial gradients...</p>
                      </div>
                   ) : gradcamImage ? (
                      <img src={gradcamImage} alt="Grad-CAM Output" className="absolute inset-0 w-full h-full object-cover" />
                   ) : (
                      <p className="text-app-textSecondary text-sm text-center">Run the model to map diagnostic focus regions.</p>
                   )}
                </div>
             </div>
           </div>
        </div>
      </section>

      {/* Mock Diagnostics Table */}
      <section>
        <h2 className="text-xl font-semibold text-app-text mb-4">Recent Diagnostics Log</h2>
        <div className="bg-white border border-app-border rounded-xl shadow-sm overflow-hidden">
           <div className="overflow-x-auto">
             <table className="w-full text-left text-sm whitespace-nowrap">
               <thead className="bg-slate-50 border-b border-app-border text-app-textSecondary">
                 <tr>
                   <th className="px-6 py-4 font-medium">Scan ID</th>
                   <th className="px-6 py-4 font-medium">Predicted Disease</th>
                   <th className="px-6 py-4 font-medium">Confidence</th>
                   <th className="px-6 py-4 font-medium">Time</th>
                   <th className="px-6 py-4 font-medium">Flag</th>
                 </tr>
               </thead>
               <tbody className="divide-y divide-app-border text-app-text">
                 {recentDiagnostics.map((log) => (
                   <tr key={log.id} className="hover:bg-slate-50 transition-colors">
                     <td className="px-6 py-4 font-medium">{log.id}</td>
                     <td className="px-6 py-4">{log.disease}</td>
                     <td className="px-6 py-4 font-semibold">{log.confidence}</td>
                     <td className="px-6 py-4 text-app-textSecondary">{log.date}</td>
                     <td className="px-6 py-4">
                        {log.status === 'Clear' ? (
                           <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-app-successBg text-app-success border border-green-200">
                             <CheckCircle2 className="w-3.5 h-3.5" /> Clear
                           </span>
                        ) : (
                           <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                             <AlertCircle className="w-3.5 h-3.5" /> Warning
                           </span>
                        )}
                     </td>
                   </tr>
                 ))}
               </tbody>
             </table>
           </div>
        </div>
      </section>

    </div>
  );
}
