import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Login from './pages/Login';
import Home from './pages/Home';
import Classification from './pages/Classification';
import Performance from './pages/Performance';
import GanImages from './pages/GanImages';
import ProjectInfo from './pages/ProjectInfo';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');

  const renderContent = () => {
    switch (activeTab) {
      case 'home': return <Home setActiveTab={setActiveTab} />;
      case 'classification': return <Classification />;
      case 'performance': return <Performance />;
      case 'gan': return <GanImages />;
      case 'info': return <ProjectInfo />;
      default: return <Home setActiveTab={setActiveTab} />;
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden font-sans text-app-text">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        <Navbar />
        <main className="flex-1 overflow-y-auto w-full p-6 lg:p-10 flex flex-col">
          <div className="max-w-6xl mx-auto w-full flex-1 animate-page-in">
            {renderContent()}
          </div>
          <Footer />
        </main>
      </div>
    </div>
  );
}
