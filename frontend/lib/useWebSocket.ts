'use client';
import { useEffect, useRef } from 'react';
import { useStore } from '@/lib/store';

const getWsUrl = () => {
  let wsUrl = process.env.NEXT_PUBLIC_WS_URL;
  if (wsUrl) {
    if (!wsUrl.endsWith('/ws')) {
      wsUrl = wsUrl.endsWith('/') ? wsUrl + 'ws' : wsUrl + '/ws';
    }
    return wsUrl;
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    let apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (apiUrl.endsWith('/')) apiUrl = apiUrl.slice(0, -1);
    
    if (apiUrl.startsWith('https://')) return apiUrl.replace('https://', 'wss://') + '/ws';
    if (apiUrl.startsWith('http://')) return apiUrl.replace('http://', 'ws://') + '/ws';
    return `wss://${apiUrl.replace('wss://', '').replace('ws://', '')}/ws`;
  }
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
  }
  return 'ws://localhost:8000/ws';
};

const WS_URL = getWsUrl();

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const {
    addLog, setLogs, setBotState, setCandles, setPrediction,
    setTAComposite, setNews, addPosition, updatePosition, setStats,
    setCurrentPrice, setWsConnected, addCandle,
  } = useStore();

  const connect = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setWsConnected(true);
      console.log('[WS] Connected');
    };

    ws.onmessage = (evt) => {
      if (wsRef.current !== ws) return;
      try {
        const raw = evt.data;
        // Skip raw ping/pong strings
        if (raw === 'pong' || raw === 'ping') return;
        const msg = JSON.parse(raw);
        const { type, data } = msg;

        switch (type) {
          case 'init':
            if (data) {
              setBotState({
                status: data.status,
                status_message: data.status_message,
                symbol: data.symbol,
                timeframe: data.timeframe,
                running: data.running,
                cycle_count: data.cycle_count,
                started_at: data.started_at,
                confidence_threshold: data.confidence_threshold,
                exchange: data.exchange,
                exchange_connected: data.exchange_connected,
                paper_mode: data.paper_mode ?? true,
                last_prediction: data.last_prediction || {},
                open_positions: data.open_positions || [],
                stats: data.stats || {},
              });
              if (data.stats) setStats(data.stats);
            }
            break;

          case 'logs':
            if (Array.isArray(data)) setLogs(data.reverse());
            break;

          case 'log':
            addLog(data);
            break;

          case 'status_update':
            setBotState({ status: data.status, status_message: data.message });
            break;

          case 'analysis_update':
            if (data.ta) setTAComposite(data.ta);
            if (data.prediction) setPrediction(data.prediction);
            if (data.news) setNews(data.news);
            if (data.candles) setCandles(data.candles);
            if (data.price) setCurrentPrice(data.price);
            break;
          
          case 'price_update':
            if (data.price) setCurrentPrice(data.price);
            if (data.candle) addCandle(data.candle);
            break;

          case 'position_opened':
            addPosition(data);
            break;

          case 'position_closed':
            updatePosition(data);
            break;

          case 'cycle_complete':
            setBotState({ cycle_count: data.cycle });
            break;

          case 'heartbeat':
            ws.send('ping');
            break;

          default:
            break;
        }
      } catch (e) {
        console.error('[WS] Parse error', e);
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setWsConnected(false);
      console.log('[WS] Disconnected — retrying in 3s');
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  };

  useEffect(() => {
    connect();
    const ping = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 25000);
    return () => {
      clearInterval(ping);
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
