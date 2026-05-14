'use client';
import { useStore } from '@/lib/store';

function ConfidenceRing({ confidence, direction }: { confidence: number; direction: string }) {
  const color = direction === 'BUY' ? 'var(--green)' : direction === 'SELL' ? 'var(--red)' : 'var(--yellow)';
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const filled = (confidence / 100) * circumference;

  return (
    <div style={{ position: 'relative', width: 140, height: 140, margin: '0 auto 16px', flexShrink: 0 }}>
      <svg width="140" height="140" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
        <circle
          cx="70" cy="70" r={radius} fill="none"
          stroke={color} strokeWidth="10"
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1)', filter: `drop-shadow(0 0 8px ${color})` }}
        />
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontSize: 28, fontWeight: 900, color, letterSpacing: '-1px' }}>{confidence.toFixed(0)}%</div>
        <div style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase' }}>Win Probability</div>
      </div>
    </div>
  );
}

export default function PredictionPanel() {
  const { prediction, taComposite, news } = useStore();

  const dir = prediction.direction || 'NEUTRAL';
  const conf = prediction.confidence || 0;
  const shouldTrade = prediction.should_trade;

  const dirConfig: Record<string, { badge: string; emoji: string; color: string }> = {
    BUY:     { badge: 'badge-green',  emoji: '🚀', color: 'var(--green)' },
    SELL:    { badge: 'badge-red',    emoji: '🔻', color: 'var(--red)' },
    NEUTRAL: { badge: 'badge-yellow', emoji: '⏸️', color: 'var(--yellow)' },
  };
  const dc = dirConfig[dir] || dirConfig.NEUTRAL;

  const scores = prediction?.component_scores;
  const newsData = (news || {}) as Record<string, unknown>;

  return (
    <div className="card animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div className="card-title">🧠 AI Prediction</div>

      <ConfidenceRing confidence={conf} direction={dir} />

      {/* Direction */}
      <div style={{ textAlign: 'center' }}>
        <span className={`badge ${dc.badge}`} style={{ fontSize: 14, padding: '5px 18px', gap: 6 }}>
          {dc.emoji} {dir}
        </span>
        <div style={{ marginTop: 8, fontSize: 12, color: shouldTrade ? 'var(--green)' : 'var(--text-muted)' }}>
          {shouldTrade ? '✅ Trade Signal Active' : `🔍 Monitoring (need ${(prediction?.threshold ?? 85).toFixed(0)}%)`}
        </div>
      </div>

      <div className="divider" />

      {/* Component scores */}
      {scores && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Score Breakdown</div>
          {[
            { label: 'ML AI Model',        value: scores.ml_ai || 50, color: 'var(--blue)' },
            { label: 'Technical Analysis', value: scores.technical, color: 'var(--primary)' },
            { label: 'News Sentiment',     value: scores.sentiment,  color: 'var(--purple)' },
            { label: 'Multi-Timeframe',    value: scores.momentum_mtf, color: 'var(--cyan)' },
            { label: 'Order Book',         value: scores.microstructure, color: 'var(--yellow)' },
          ].map((s) => (
            <div key={s.label} style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 12 }}>
                <span style={{ color: 'var(--text-secondary)' }}>{s.label}</span>
                <span style={{ color: s.color, fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>{(s.value ?? 50).toFixed(0)}%</span>
              </div>
              <div style={{ height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ width: `${s.value}%`, height: '100%', background: s.color, borderRadius: 3, transition: 'width 0.8s ease', boxShadow: `0 0 8px ${s.color}40` }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* TA composite */}
      {!!taComposite && (
        <>
          <div className="divider" />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span style={{ color: 'var(--green)' }}>▲ Bull: {taComposite.bull_signals}</span>
            <span style={{ color: 'var(--text-muted)' }}>Total: {taComposite.total_signals} signals</span>
            <span style={{ color: 'var(--red)' }}>▼ Bear: {taComposite.bear_signals}</span>
          </div>
        </>
      )}

      {/* Reasoning */}
      {prediction?.reasoning && prediction.reasoning.length > 0 && (
        <>
          <div className="divider" />
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>Reasoning</div>
            {prediction?.reasoning?.map((r, i) => (
              <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 5, display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                <span style={{ color: 'var(--primary)', flexShrink: 0 }}>›</span>
                {r}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Market Logic Verdict (New Section) */}
      <div className="divider" />
      <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 12, padding: 12, border: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ fontSize: 11, color: 'var(--primary)', marginBottom: 8, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em' }}>💎 Market Logic & Expert Verdict</div>
          <p style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6, margin: 0, fontWeight: 500 }}>
              {prediction?.expert_logic || "Analyzing market microstructure... Synthesis in progress."}
          </p>
      </div>

      {/* News sentiment */}
      {!!newsData.sentiment && (
        <>
          <div className="divider" />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Market Sentiment</span>
            <span className={`badge ${newsData.sentiment === 'POSITIVE' ? 'badge-green' : newsData.sentiment === 'NEGATIVE' ? 'badge-red' : 'badge-yellow'}`}>
              {String(newsData.sentiment)}
            </span>
          </div>
          {!!newsData.fear_greed && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: 'var(--text-muted)' }}>Fear & Greed</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace' }}>
                {String((newsData.fear_greed as Record<string,unknown>).value)} — {String((newsData.fear_greed as Record<string,unknown>).classification)}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
