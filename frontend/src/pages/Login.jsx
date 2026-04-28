import React, { useState } from 'react';
import { Leaf, Lock, Mail, ArrowRight } from 'lucide-react';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    // Simulate network request
    setTimeout(() => {
      setIsLoading(false);
      onLogin(); // Mock successful auth
    }, 800);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 relative overflow-hidden text-app-text font-sans">
      {/* Decorative background vectors */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-green-50 via-slate-50 to-slate-50"></div>
      
      <div className="z-10 w-full max-w-md p-6 sm:p-10 animate-page-in">
        <div className="bg-white rounded-3xl shadow-xl shadow-app-primaryLight/20 border border-app-border p-8 md:p-10">
          
          <div className="flex flex-col items-center mb-8">
            <div className="w-14 h-14 bg-app-primaryLight rounded-2xl flex items-center justify-center text-app-primary mb-4 shadow-inner">
               <Leaf className="w-7 h-7" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-app-text text-center">
              Welcome back to <span className="text-app-primary">TomatoDetect</span>
            </h1>
            <p className="text-sm text-app-textSecondary mt-2 text-center">
              Enter your credentials to access the analytics dashboard.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-app-textSecondary mb-1.5" htmlFor="email">Email Address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                  <Mail className="h-4 w-4 text-slate-400" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-200 rounded-xl focus:ring-2 focus:ring-app-primaryLight focus:border-app-primary bg-slate-50 focus:bg-white transition-colors text-sm text-app-text placeholder:text-slate-400"
                  placeholder="admin@tomatodetect.ai"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-app-textSecondary mb-1.5" htmlFor="password">Password</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                  <Lock className="h-4 w-4 text-slate-400" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-200 rounded-xl focus:ring-2 focus:ring-app-primaryLight focus:border-app-primary bg-slate-50 focus:bg-white transition-colors text-sm text-app-text placeholder:text-slate-400"
                  placeholder="••••••••"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading || !email || !password}
              className={`w-full py-3 mt-4 rounded-xl font-medium flex justify-center items-center gap-2 transition-all shadow-lg ${
                isLoading || !email || !password
                  ? 'bg-slate-100 text-slate-400 border border-slate-200 shadow-none cursor-not-allowed'
                  : 'bg-app-primary text-white hover:bg-app-primaryHover shadow-app-primaryLight active:scale-[0.98]'
              }`}
            >
              {isLoading ? (
                <div className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white animate-spin"></div>
              ) : (
                <>Sign In <ArrowRight className="w-4 h-4 ml-1" /></>
              )}
            </button>
          </form>
          
        </div>
        
        <div className="mt-8 text-center text-sm text-slate-500">
           Protected by state-of-the-art encryption.
        </div>
      </div>
    </div>
  );
}
