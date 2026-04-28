import React from 'react';
import { AlertCircle, CheckCircle2, AlertTriangle, Info } from 'lucide-react';

/**
 * Standardized Alert Component 
 * Types restricted to minimal allowed options for consistency.
 */
export default function Alert({ type = 'info', title, message }) {
  const styles = {
    error: {
      container: 'bg-app-errorLight border-red-200 text-app-error',
      icon: <AlertCircle className="w-5 h-5 shrink-0" />,
    },
    success: {
      container: 'bg-app-successLight border-green-200 text-app-success',
      icon: <CheckCircle2 className="w-5 h-5 shrink-0" />,
    },
    warning: {
      container: 'bg-app-warningLight border-amber-200 text-app-warning',
      icon: <AlertTriangle className="w-5 h-5 shrink-0" />,
    },
    info: {
      container: 'bg-app-primaryLight border-blue-200 text-app-primary',
      icon: <Info className="w-5 h-5 shrink-0" />,
    }
  };

  const config = styles[type] || styles.info;

  return (
    <div className={`flex items-start gap-3 border rounded-lg px-4 py-3 ${config.container}`}>
      <div className="mt-0.5">{config.icon}</div>
      <div className="flex flex-col">
        {title && <span className="font-medium text-sm">{title}</span>}
        <span className="text-sm opacity-90">{message}</span>
      </div>
    </div>
  );
}
