import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, Target, CheckCircle, Crosshair, RefreshCw } from 'lucide-react';

export default function Performance() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const API_BASE = import.meta.env.VITE_API_URL || 'https://tomato-leaf-disease-2-xb8w.onrender.com/';

  useEffect(() => {
    axios.get(`${API_BASE}/metrics/`)
      .then(res => {
        setMetrics(res.data);
      })
      .catch(err => console.error("Metrics failed to load"))
      .finally(() => setLoading(false));
  }, [API_BASE]);

  const MetricCard = ({ title, value, icon: Icon, colorClass }) => (
    <div className="bg-white border border-app-border rounded-2xl p-6 flex flex-col justify-between min-h-[140px] relative overflow-hidden group shadow-sm hover:shadow-md transition-shadow">
      <div className="absolute top-0 right-0 p-4 opacity-[0.06] group-hover:opacity-[0.12] transition-opacity">
        <Icon className="w-24 h-24 text-app-primary" />
      </div>
      <div>
        <p className="text-sm font-medium text-app-textSecondary">{title}</p>
        <p className={`text-4xl font-semibold mt-3 ${colorClass}`}>
          {value ? `${(value * 100).toFixed(2)}%` : '---'}
        </p>
      </div>
      <div className="absolute bottom-0 left-0 h-1 bg-slate-100 w-full group-hover:bg-app-primary transition-colors" />
    </div>
  );

  return (
    <div className="space-y-8 animate-page-in">
      <div>
        <h2 className="text-3xl font-semibold text-app-text mb-2">
          Model Validation Metrics
        </h2>
        <p className="text-app-textSecondary">Performance benchmarks from the trained EfficientNet classifier.</p>
      </div>

      {loading ? (
        <div className="flex justify-center items-center h-40 bg-white border border-app-border rounded-2xl">
          <RefreshCw className="w-8 h-8 animate-spin text-app-textSecondary" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard title="Accuracy Rating" value={metrics?.test_accuracy} icon={Target} colorClass="text-app-text" />
            <MetricCard title="Precision Factor" value={metrics?.test_precision} icon={Crosshair} colorClass="text-app-text" />
            <MetricCard title="Recall Index" value={metrics?.test_recall} icon={Activity} colorClass="text-app-text" />
            <MetricCard title="F1 Score" value={metrics?.test_f1} icon={CheckCircle} colorClass="text-app-primary" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white border border-app-border rounded-2xl flex flex-col p-6 shadow-sm">
              <div className="text-sm font-medium text-app-textSecondary mb-4">
                 Training & Validation Loss
              </div>
              <div className="flex-1 bg-slate-50 rounded-xl border border-app-border flex justify-center items-center overflow-hidden min-h-[400px] p-4">
                <img 
                  src={`${API_BASE}/assets/training_curves.png`}
                  alt="Training Curves" 
                  className="max-h-full object-contain"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
            </div>

            <div className="bg-white border border-app-border rounded-2xl flex flex-col p-6 shadow-sm">
              <div className="text-sm font-medium text-app-textSecondary mb-4">
                 Final Confusion Matrix
              </div>
              <div className="flex-1 bg-slate-50 rounded-xl border border-app-border flex justify-center items-center overflow-hidden min-h-[400px] p-4">
                <img 
                  src={`${API_BASE}/assets/confusion_matrix.png`}
                  alt="Confusion Matrix" 
                  className="max-h-full object-contain"
                  onError={(e) => e.target.style.display = 'none'}
                />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
