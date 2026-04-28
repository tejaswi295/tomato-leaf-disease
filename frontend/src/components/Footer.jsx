import React from 'react';
import { Leaf } from 'lucide-react';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-16 pt-8 border-t border-slate-200 bg-transparent text-sm text-app-textSecondary pb-6">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Leaf className="w-5 h-5 text-app-primary" />
          <span className="text-lg font-bold tracking-tight text-app-text">TomatoDetect</span>
        </div>
        <p className="text-center">© {currentYear} TomatoDetect Analytics. All rights reserved.</p>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 block"></span>
        </div>
      </div>
    </footer>
  );
}
