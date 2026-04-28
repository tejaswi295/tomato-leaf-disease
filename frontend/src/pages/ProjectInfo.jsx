import React from 'react';
import { Database, Shield, Zap, Cpu } from 'lucide-react';

export default function ProjectInfo() {
  const specs = [
    { label: 'Core Engine', value: 'PyTorch / Torchvision', icon: Cpu },
    { label: 'Neural Backbone', value: 'EfficientNet-B0', icon: Zap },
    { label: 'Training Dataset', value: 'PlantVillage (10 Classes)', icon: Database },
    { label: 'Optimization', value: 'EarlyStopping / LR Decay', icon: Shield },
  ];

  return (
    <div className="space-y-8 animate-page-in max-w-4xl">
      <div>
        <h2 className="text-3xl font-semibold text-app-text mb-2">
          System Overview
        </h2>
        <p className="text-app-textSecondary">Technical specifications and supported disease classes.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {specs.map((s, idx) => {
          const Icon = s.icon;
          return (
            <div key={idx} className="bg-white border border-app-border rounded-2xl p-6 flex flex-col justify-center relative overflow-hidden group shadow-sm hover:shadow-md transition-shadow">
              <Icon className="absolute top-1/2 -translate-y-1/2 right-6 w-16 h-16 opacity-[0.06] group-hover:opacity-[0.12] group-hover:scale-110 transition-all text-app-primary" />
              <div className="text-sm font-medium text-app-primary mb-1">{s.label}</div>
              <div className="text-2xl font-semibold text-app-text relative z-10">
                {s.value}
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-white border border-app-border rounded-2xl p-8 shadow-sm">
        <h3 className="font-semibold text-xl text-app-text mb-6 flex items-center gap-3">
          <span className="w-1 h-6 bg-app-primary rounded-sm inline-block" /> Data Dictionary (Classes)
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-y-4 gap-x-8">
          {[
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites',
            'Tomato___Target_Spot',
            'Tomato___Yellow_Leaf_Curl_Virus',
            'Tomato___mosaic_virus',
            'Tomato___healthy'
          ].map((item, idx) => (
            <div key={idx} className="flex items-center gap-3 group">
              <span className="text-app-primary font-mono text-sm bg-app-primaryLight px-2 py-0.5 rounded group-hover:bg-blue-100 transition-colors">{(idx + 1).toString().padStart(2, '0')}</span>
              <span className="text-app-text font-medium">
                {item.replace('Tomato___', '').replace(/_/g, ' ')}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
