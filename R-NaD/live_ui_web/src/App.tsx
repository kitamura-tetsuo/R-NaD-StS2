import React, { useState, useEffect, useRef } from 'react';
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid, ReferenceLine, Label, Tooltip, XAxis, YAxis, Legend
} from 'recharts';
import {
  Heart, Layers, Shield,
  ChevronRight, Activity, TrendingUp
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Card {
  id: string;
  name: string;
  type?: string;
  cost: number;
  is_generated?: boolean;
}

interface Enemy {
  name: string;
  hp: number;
  maxHp: number;
  block: number;
  intents?: any[];
}

interface GameState {
  type: string;
  floor: number;
  player: {
    hp: number;
    maxHp: number;
    energy: number;
    block: number;
  };
  enemies: Enemy[];
  hand: Card[];
  drawPile: Card[];
  discardPile: Card[];
  exhaustPile: Card[];
}

interface ActionProb {
  id: number;
  name: string;
  prob: number;
  isSelected?: boolean;
  searchType?: string;
  isStochastic?: boolean;
}

interface LiveData {
  state: GameState;
  predicted_v: number;
  reward: number;
  cum_reward: number;
  top_actions: ActionProb[];
  timestamp: number;
  action_idx: number;
  terminal: boolean;
  reset?: boolean;
  step_id?: number;
  events?: { time: number; label: string; color: string }[];
  stochastic_override?: boolean;
}

const App: React.FC = () => {
  const [data, setData] = useState<LiveData | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [actionHistory, setActionHistory] = useState<ActionProb[][]>([]);
  const [events, setEvents] = useState<{ time: number; label: string; color: string }[]>([]);
  const [connected, setConnected] = useState(false);

  const prevStateRef = useRef<GameState | null>(null);
  const prevActionIdxRef = useRef<number | null>(null);
  const prevStepIdRef = useRef<number | null>(null);
  const vMinRef = useRef<number>(Infinity);
  const vMaxRef = useRef<number>(-Infinity);
  const globalCounterRef = useRef<number>(0);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`ws://${window.location.hostname}:8051/ws`);

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000);
      };

      ws.onmessage = (event) => {
        const newData: LiveData = JSON.parse(event.data);
        const { timestamp, predicted_v, state, reset } = newData;

        setData(newData);

        // Handle Trajectory/Retry Reset
        if (vMinRef.current === Infinity || reset || (prevStateRef.current && newData.state.floor < prevStateRef.current.floor)) {
          vMinRef.current = Math.min(predicted_v, newData.cum_reward) - 0.1;
          vMaxRef.current = Math.max(predicted_v, newData.cum_reward) + 0.1;
          if (reset) {
            setHistory([]);
            setEvents([]);
            setActionHistory([]);
            prevActionIdxRef.current = null;
            prevStepIdRef.current = null;
            globalCounterRef.current = 0;
          }
        } else {
          vMinRef.current = Math.min(vMinRef.current, predicted_v, newData.cum_reward);
          vMaxRef.current = Math.max(vMaxRef.current, predicted_v, newData.cum_reward);
        }

        if (newData.events) {
          setEvents(newData.events);
        }

        if (newData.top_actions) {
          const isNewStep = newData.step_id !== undefined && newData.step_id !== prevStepIdRef.current;
          const isNewAction = newData.action_idx !== prevActionIdxRef.current;
          const isNewState = prevStateRef.current?.type !== newData.state.type || prevStateRef.current?.floor !== newData.state.floor;

          if (isNewStep || isNewAction || isNewState) {
            setActionHistory(prev => {
              const updated = [newData.top_actions, ...prev];
              return updated.slice(0, 5);
            });
            if (newData.step_id !== undefined) prevStepIdRef.current = newData.step_id;
            prevActionIdxRef.current = newData.action_idx;
          } else {
            setActionHistory(prev => {
              const updated = [...prev];
              updated[0] = newData.top_actions;
              return updated;
            });
          }
        }

        setHistory(prev => {
          globalCounterRef.current += 1;
          const point: any = {
            time: timestamp,
            idx: globalCounterRef.current,
            v: predicted_v,
            reward: newData.reward,
            cumReward: newData.cum_reward,
            playerHp: state.player?.hp || 0,
            playerBlock: state.player?.block || 0,
            incoming: (newData as any).state?.predicted_total_damage || 0
          };

          state.enemies?.forEach((enemy, idx) => {
            point[`enemy${idx}`] = enemy.hp;
          });

          const updated = [...prev, point];
          return updated.slice(-50);
        });

        prevStateRef.current = newData.state;
      };
    };

    connect();
  }, []);

  if (!data) return (
    <div className="flex h-screen items-center justify-center bg-bg-dark text-white font-sans">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-brand-primary border-t-transparent" />
        <h2 className="text-xl font-medium tracking-tight">Initializing Neuromancer-StS2 Bridge...</h2>
        <p className="text-sm opacity-50">Waiting for local game state broadcast</p>
      </div>
    </div>
  );

  const { state } = data;

  return (
    <div className="min-h-screen bg-bg-dark text-gray-100 p-4 font-sans selection:bg-brand-primary/30">
      <main className="grid grid-cols-10 gap-2 items-start">
        {/* Left: Trend Charts - 8 columns (2/3) */}
        <div className="col-span-6 space-y-6">
          <div className="flex flex-col gap-6">
            {/* Top Chart: V-Value Trend */}
            <section className="glass p-4 min-h-[260px]">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 flex items-center gap-2">
                <TrendingUp className="w-3 h-3" /> V-Value Trend
              </h3>
              <div className="h-36 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} syncId="sts2-charts">
                    <Line
                      type="linear"
                      dataKey="v"
                      stroke="#00ffcc"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      name="Value (V)"
                    />

                    <Line
                      type="linear"
                      dataKey="cumReward"
                      stroke="#facc15"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                      name="Cum. Reward"
                    />
                    <XAxis dataKey="idx" hide />
                    <YAxis domain={[vMinRef.current - 0.05, vMaxRef.current + 0.05]} hide />
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e202d', border: 'none', borderRadius: '8px', fontSize: '10px' }}
                      itemStyle={{ color: '#00ffcc' }}
                    />
                    <Legend verticalAlign="top" align="right" iconSize={10} wrapperStyle={{ fontSize: '10px', paddingBottom: '30px' }} />
                    {events.map((ev, idx) => {
                      const matchedPoint = history.find(p => p.time >= ev.time);
                      if (!matchedPoint || ev.time < history[0].time) return null;
                      return (
                        <ReferenceLine key={idx} x={matchedPoint.idx} stroke={ev.color} strokeDasharray="4 4">
                          <Label value={ev.label} position="top" fill={ev.color} fontSize={8} fontWeight="bold" offset={5} />
                        </ReferenceLine>
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Middle Chart: HP Trend */}
            <section className="glass p-4 min-h-[220px]">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 flex items-center gap-2">
                <Heart className="w-3 h-3 text-red-500" /> Player & Enemy Health
              </h3>
              <div className="h-36 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} syncId="sts2-charts">
                    <XAxis dataKey="idx" hide />
                    <YAxis domain={[0, 'auto']} hide />
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e202d', border: 'none', borderRadius: '8px', fontSize: '10px' }}
                    />
                    <Line
                      type="stepAfter"
                      dataKey="playerHp"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      name="Player HP"
                    />
                    {[0, 1, 2, 3, 4].map(idx => (
                      <Line
                        key={idx}
                        type="stepAfter"
                        dataKey={`enemy${idx}`}
                        stroke="#a855f7"
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                        name="Enemy HP"
                        legendType={idx === 0 ? 'line' : 'none'}
                      />
                    ))}
                    <Legend verticalAlign="top" align="right" iconSize={10} wrapperStyle={{ fontSize: '10px', paddingBottom: '30px' }} />
                    {events.map((ev, idx) => {
                      const matchedPoint = history.find(p => p.time >= ev.time);
                      if (!matchedPoint || ev.time < history[0].time) return null;
                      return (
                        <ReferenceLine key={idx} x={matchedPoint.idx} stroke={ev.color} strokeDasharray="4 4">
                           <Label value={ev.label} position="top" fill={ev.color} fontSize={8} fontWeight="bold" offset={5} />
                        </ReferenceLine>
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Bottom Chart: Defense Trend */}
            <section className="glass p-4 min-h-[220px]">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2 flex items-center gap-2">
                <Shield className="w-3 h-3 text-blue-400" /> Defense & Incoming Damage
              </h3>
              <div className="h-36 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} syncId="sts2-charts">
                    <XAxis dataKey="idx" hide />
                    <YAxis hide />
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e202d', border: 'none', borderRadius: '8px', fontSize: '10px' }}
                    />
                    <Line
                      type="stepAfter"
                      dataKey="playerBlock"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      name="Player Block"
                    />
                    <Line
                      type="stepAfter"
                      dataKey="incoming"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                      name="Incoming Damage"
                    />
                    <Legend verticalAlign="top" align="right" iconSize={10} wrapperStyle={{ fontSize: '10px', paddingBottom: '30px' }} />
                    {events.map((ev, idx) => {
                      const matchedPoint = history.find(p => p.time >= ev.time);
                      if (!matchedPoint || ev.time < history[0].time) return null;
                      return (
                        <ReferenceLine key={idx} x={matchedPoint.idx} stroke={ev.color} strokeDasharray="4 4">
                           <Label value={ev.label} position="top" fill={ev.color} fontSize={8} fontWeight="bold" offset={5} />
                        </ReferenceLine>
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </div>

        {/* Right: Card Piles (Vertical) - 2 columns */}
        <aside className="col-span-2 space-y-4 max-h-[calc(100vh-140px)] overflow-y-auto pr-2 custom-scrollbar">
          <PileSection
            title="Hand"
            cards={state.hand || []}
            icon={<Layers className="w-3 h-3 text-indigo-400" />}
            color="indigo"
            current
          />
          <PileSection
            title="Draw"
            cards={state.drawPile || []}
            icon={<ChevronRight className="rotate-270 w-3 h-3 text-gray-400" />}
            color="gray"
          />
          <PileSection
            title="Discard"
            cards={state.discardPile || []}
            icon={<ChevronRight className="rotate-90 w-3 h-3 text-gray-400" />}
            color="gray"
          />
        </aside>

        {/* Far Right: AI Action History Pipeline - 2 columns */}
        <aside className="col-span-2 space-y-4">
          <div className="flex items-center justify-between px-2 mb-2">
            <h3 className="text-[10px] font-bold uppercase tracking-wider text-gray-500 flex items-center gap-2">
              <Activity className="w-4 h-4 text-brand-secondary" /> Action Timeline
            </h3>
            {data.stochastic_override && (
              <span className="px-1.5 py-0.5 bg-orange-600 text-white text-[8px] font-black tracking-tighter uppercase rounded animate-pulse shadow-[0_0_8px_rgba(234,88,12,0.4)] border border-orange-400/50">
                RETRY STOCHASTIC OVERRIDE
              </span>
            )}
          </div>
          <div className="space-y-4">
            {actionHistory.map((actions, stepIdx) => (
              <div
                key={`${stepIdx}-${actions[0]?.id}`}
                className={cn(
                  "glass p-2 relative overflow-hidden transition-all duration-300",
                  stepIdx === 0 && "border-brand-secondary/40 shadow-[0_0_15px_rgba(0,255,204,0.1)] ring-1 ring-brand-secondary/10"
                )}
              >
                {stepIdx === 0 && (
                  <div className="absolute top-0 right-0 px-1.5 py-0.5 bg-brand-secondary text-bg-dark text-[8px] font-black tracking-tighter uppercase rounded-bl">
                    LIVE
                  </div>
                )}
                <div className="space-y-3">
                  {actions.map((action) => (
                    <div key={action.id} className={cn(
                      "space-y-0.5",
                      action.isSelected ? "opacity-100" : "opacity-75"
                    )}>
                      <div className="flex justify-between items-end leading-none">
                        <span className={cn(
                          "text-[10px] truncate max-w-[80%]",
                          action.isSelected ? "text-brand-secondary font-bold" : "text-gray-200 font-medium"
                        )}>
                          {action.isSelected ? '▶ ' : ''}{action.name}
                          {action.searchType && action.isSelected && (
                            <span className={cn(
                              "ml-1 px-1 text-bg-dark text-[7px] font-black rounded h-fit self-center uppercase",
                              action.searchType === 'lethal' ? "bg-red-500" : "bg-brand-secondary"
                            )}>
                              {action.searchType}
                            </span>
                          )}
                          {action.isStochastic && action.isSelected && (
                            <span className="ml-1 px-1 bg-orange-600 text-white text-[7px] font-black rounded h-fit self-center uppercase shadow-[0_0_4px_rgba(234,88,12,0.4)]">
                              STOCHASTIC
                            </span>
                          )}
                        </span>
                        <span className={cn(
                          "text-[8px] font-mono",
                          action.isSelected ? "text-brand-secondary font-bold" : "text-gray-400"
                        )}>
                          {(() => {
                            const p = action.prob * 100;
                            if (p > 0 && p < 0.1) return '<0.1%';
                            if (p > 99.9 && p < 100) return '>99.9%';
                            return p.toFixed(1) + '%';
                          })()}
                        </span>
                      </div>
                      <div className="h-1 w-full bg-gray-800/50 rounded-full overflow-hidden">
                        <div
                          className={cn(
                            "h-full rounded-full",
                            action.isSelected ? "bg-brand-secondary shadow-[0_0_4px_rgba(0,255,204,0.3)]" : "bg-gray-600"
                          )}
                          style={{ width: `${action.prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            {actionHistory.length === 0 && (
              <div className="text-[10px] text-center text-gray-600 italic py-10">
                Waiting for AI decisions...
              </div>
            )}
          </div>
        </aside>
      </main>

      {!connected && (
        <div className="fixed bottom-4 right-4 bg-red-900/80 text-white text-xs px-3 py-1.5 rounded-full border border-red-500 backdrop-blur-sm animate-pulse">
          Connection Lost - Reconnecting...
        </div>
      )}
    </div>
  );
};


const PileSection = ({ title, cards, icon, current }: any) => {
  // If it's a list of strings, convert to objects
  const processedCards = cards.map((c: any) => typeof c === 'string' ? { name: c, id: c } : c);

  const typeOrder: Record<string, number> = {
    'Attack': 1,
    'Skill': 2,
    'Power': 3
  };

  const sortedCards = [...processedCards].sort((a, b) => {
    const orderA = typeOrder[a.type] || 99;
    const orderB = typeOrder[b.type] || 99;
    if (orderA !== orderB) return orderA - orderB;
    return a.name.localeCompare(b.name);
  });

  return (
    <div className={cn(
      "glass overflow-hidden flex flex-col transition-all duration-300",
      current ? "border-brand-secondary/20 shadow-[0_0_20px_rgba(0,255,204,0.05)]" : "opacity-80 hover:opacity-100"
    )}>
      <div className="p-3 border-b border-white/5 flex justify-between items-center bg-white/5">
        <h4 className="text-xs font-bold uppercase tracking-wider flex items-center gap-2">
          {icon} {title} ({cards.length})
        </h4>
      </div>
      <div className="p-2 space-y-1.5 overflow-visible">
        {sortedCards.length === 0 ? (
          <div className="text-[10px] text-gray-600 italic py-2 text-center">Empty</div>
        ) : (
          sortedCards.map((card: any, idx: number) => (
            <div key={`${card.id}-${idx}`} className={cn(
              "text-xs px-1 py-1 flex justify-between group hover:bg-white/5 transition-colors border-b border-white/5 last:border-0",
              card.is_generated && "text-brand-secondary/80 font-medium"
            )}>
              <span className="font-medium truncate pr-2">{card.name}</span>
              <span className="opacity-40 font-mono text-[10px] shrink-0">{card.cost}E</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default App;
