import React from 'react';

/**
 * Metric card component for high-level statistics and indicators.
 */
export default function StatCard({ title, value, subtitle, tag, tagType = 'demo', icon }) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <span className="stat-card-title">{title}</span>
        {icon && (
          <div className="stat-card-icon" style={{ backgroundColor: '#f1f5f9', color: '#0f172a' }}>
            {icon}
          </div>
        )}
      </div>

      <div className="stat-card-value">{value}</div>

      {subtitle && <div className="stat-card-subtitle">{subtitle}</div>}

      {tag && (
        <span className={`stat-card-tag tag-${tagType}`}>
          {tag}
        </span>
      )}
    </div>
  );
}
