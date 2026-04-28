import React, { useState } from 'react';
import { 
  LayoutDashboard, 
  Scan, 
  BarChart3, 
  ImagePlus, 
  Info, 
  PanelLeftClose, 
  PanelLeftOpen,
  Leaf
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const [expanded, setExpanded] = useState(true);

  const navItems = [
    { id: 'home', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'classification', label: 'Diagnostic Scanner', icon: Scan },
    { id: 'performance', label: 'Metrics', icon: BarChart3 },
    { id: 'gan', label: 'GAN Data', icon: ImagePlus },
    { id: 'info', label: 'System Info', icon: Info },
  ];

  return (
    <aside className={`h-screen bg-white border-r border-app-border flex flex-col transition-all duration-300 ${expanded ? 'w-64' : 'w-20'} shrink-0 z-50 relative`}>
      <div className={`h-16 flex items-center border-b border-app-border ${expanded ? 'px-6 justify-between' : 'justify-center'} shrink-0`}>
        {expanded ? (
          <div className="flex items-center tracking-tight text-xl font-bold select-none cursor-pointer" onClick={() => setActiveTab('home')}>
            <span className="text-app-text">Tomato</span>
            <span className="text-app-primary">Detect</span>
          </div>
        ) : (
          <div className="text-app-primary cursor-pointer" onClick={() => setActiveTab('home')}>
            <Leaf className="w-6 h-6" />
          </div>
        )}
      </div>

      <button 
        onClick={() => setExpanded(!expanded)}
        className="absolute -right-3 top-20 bg-white border border-app-border rounded-full p-1 text-app-textSecondary hover:text-app-primary shadow-sm hover:shadow transition-colors z-[60] cursor-pointer"
      >
        {expanded ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
      </button>

      <nav className="flex-1 py-6 px-3 space-y-2 overflow-y-auto overflow-x-hidden">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              title={expanded ? undefined : item.label}
              className={`w-full flex items-center p-3 rounded-lg transition-colors cursor-pointer active:scale-95 border ${isActive ? 'bg-app-primaryLight text-app-primary border-blue-200 shadow-sm' : 'border-transparent text-app-textSecondary hover:bg-slate-50 hover:text-app-text'}`}
            >
              <Icon className={`w-5 h-5 shrink-0 ${expanded ? 'mr-3' : ''}`} />
              {expanded && <span className="font-medium text-sm whitespace-nowrap">{item.label}</span>}
            </button>
          );
        })}
      </nav>

    </aside>
  );
}
