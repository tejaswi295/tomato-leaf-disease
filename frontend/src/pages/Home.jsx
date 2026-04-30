import React from 'react';
import { 
  Activity, 
  LayoutTemplate,
  CheckCircle2,
  AlertCircle,
  Leaf
} from 'lucide-react';

export default function Home({ setActiveTab }) {

  const recentDiagnostics = [
    { id: 'SCN-8092', disease: 'Tomato Early blight', confidence: '99.2%', date: '2 Mins Ago', status: 'Warning' },
    { id: 'SCN-8091', disease: 'Tomato Healthy', confidence: '98.8%', date: '15 Mins Ago', status: 'Clear' },
    { id: 'SCN-8090', disease: 'Tomato Leaf Mold', confidence: '94.5%', date: '1 Hour Ago', status: 'Warning' },
    { id: 'SCN-8089', disease: 'Tomato Target Spot', confidence: '89.1%', date: '3 Hours Ago', status: 'Warning' },
  ];

  return (
    <div className="animate-page-in space-y-8">

      {/* Header */}
      <section className="bg-white border border-app-border rounded-2xl p-8 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <Leaf className="w-6 h-6 text-green-600" />
          <h1 className="text-3xl font-semibold text-app-text">
            Tomato Disease Detection System
          </h1>
        </div>

        <p className="text-app-textSecondary max-w-2xl">
          Upload a tomato leaf image to detect diseases using a deep learning model.
          The system identifies plant health conditions with high accuracy and speed.
        </p>

        <button
          onClick={() => setActiveTab('classification')}
          className="mt-6 px-6 py-3 rounded-xl bg-app-primary text-white font-medium hover:bg-app-primaryHover shadow-md active:scale-95"
        >
          Start Diagnosis
        </button>
      </section>

      {/* Info Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">

        <div className="bg-white border border-app-border rounded-xl p-5 shadow-sm">
          <div className="text-sm text-app-textSecondary mb-1">Model</div>
          <div className="text-lg font-semibold text-app-text">EfficientNet-B0</div>
        </div>

        <div className="bg-white border border-app-border rounded-xl p-5 shadow-sm">
          <div className="text-sm text-app-textSecondary mb-1">Classes</div>
          <div className="text-lg font-semibold text-app-text">10 Diseases</div>
        </div>

        <div className="bg-white border border-app-border rounded-xl p-5 shadow-sm">
          <div className="text-sm text-app-textSecondary mb-1">Accuracy</div>
          <div className="text-lg font-semibold text-app-text">~98%</div>
        </div>

      </section>

      {/* Quick Actions */}
      <section className="bg-white border border-app-border rounded-2xl p-6 shadow-sm">
        <h2 className="text-xl font-semibold text-app-text mb-4 flex items-center gap-2">
          <LayoutTemplate className="w-5 h-5 text-app-primary" />
          Quick Actions
        </h2>

        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => setActiveTab('classification')}
            className="px-5 py-3 rounded-xl bg-app-primary text-white font-medium hover:bg-app-primaryHover shadow-md"
          >
            Run Diagnosis
          </button>

          <button
            onClick={() => setActiveTab('metrics')}
            className="px-5 py-3 rounded-xl border border-app-border text-app-text hover:bg-slate-50"
          >
            View Metrics
          </button>
        </div>
      </section>

      {/* Diagnostics Table */}
      <section>
        <h2 className="text-xl font-semibold text-app-text mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-app-primary" />
          Recent Diagnostics Log
        </h2>

        <div className="bg-white border border-app-border rounded-xl shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm whitespace-nowrap">
              <thead className="bg-slate-50 border-b border-app-border text-app-textSecondary">
                <tr>
                  <th className="px-6 py-4 font-medium">Scan ID</th>
                  <th className="px-6 py-4 font-medium">Disease</th>
                  <th className="px-6 py-4 font-medium">Confidence</th>
                  <th className="px-6 py-4 font-medium">Time</th>
                  <th className="px-6 py-4 font-medium">Status</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-app-border text-app-text">
                {recentDiagnostics.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50">
                    <td className="px-6 py-4 font-medium">{log.id}</td>
                    <td className="px-6 py-4">{log.disease}</td>
                    <td className="px-6 py-4 font-semibold">{log.confidence}</td>
                    <td className="px-6 py-4 text-app-textSecondary">{log.date}</td>
                    <td className="px-6 py-4">
                      {log.status === 'Clear' ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-green-50 text-green-700 border border-green-200">
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
