import React, { useState, useEffect, useRef } from 'react';
import { 
  ResponsiveContainer, LineChart, Line, CartesianGrid, ReferenceLine, Label, Tooltip
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Heart, Sword, Shield, Zap, Layers, Trash2, 
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
}

interface LiveData {
  state: GameState;
  predicted_v: number;
  top_actions: ActionProb[];
  timestamp: number;
  action_idx: number;
}

const App: React.FC = () => {
  const [data, setData] = useState<LiveData | null>(null);
  const [vHistory, setVHistory] = useState<{ time: number; v: number }[]>([]);
  const [events, setEvents] = useState<{ time: number; label: string; color: string }[]>([]);
  const [connected, setConnected] = useState(false);
  
  const prevStateRef = useRef<GameState | null>(null);
  const prevActionIdxRef = useRef<number | null>(null);
  const turnCounterRef = useRef<number>(1);

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
        const { timestamp, predicted_v } = newData;
        
        setData(newData);
        
        if (newData.events) {
          setEvents(newData.events);
        }
        
        setVHistory(prev => {
          const updated = [...prev, { time: timestamp, v: predicted_v }];
          return updated.slice(-50);
        });
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

  const { state, predicted_v, top_actions } = data;

  return (
    <div className="min-h-screen bg-bg-dark text-gray-100 p-4 font-sans selection:bg-brand-primary/30">
      {/* Header Stat Bar */}
      <header className="grid grid-cols-5 gap-4 mb-6">
        <StatCard 
          label="Floor" 
          value={state.floor} 
          icon={<ChevronRight className="w-4 h-4 text-brand-primary" />} 
          variant="normal" 
        />
        <StatCard 
          label="Health" 
          value={`${state.player.hp}/${state.player.maxHp}`} 
          icon={<Heart className="w-4 h-4 text-red-500" />} 
          percent={(state.player.hp / state.player.maxHp) * 100}
          variant="health"
        />
        <StatCard 
          label="Energy" 
          value={state.player.energy} 
          icon={<Zap className="w-4 h-4 text-yellow-400" />} 
          variant="energy"
        />
        <StatCard 
          label="Block" 
          value={state.player.block} 
          icon={<Shield className="w-4 h-4 text-blue-400" />} 
          variant="block"
        />
        <StatCard 
          label="Value (V)" 
          value={predicted_v.toFixed(3)} 
          icon={<Activity className="w-4 h-4 text-brand-secondary" />} 
          variant="vvalue"
        />
      </header>

      <main className="grid grid-cols-12 gap-6 items-start">
        {/* Left: Card Piles (Vertical) */}
        <aside className="col-span-3 space-y-6">
          <PileSection 
            title="Hand" 
            cards={state.hand} 
            icon={<Layers className="w-4 h-4 text-indigo-400" />}
            color="indigo"
            current
          />
          <PileSection 
            title="Draw Pile" 
            cards={state.drawPile} 
            icon={<ChevronRight className="rotate-270 w-4 h-4 text-gray-400" />}
            color="gray"
          />
          <PileSection 
            title="Discard Pile" 
            cards={state.discardPile} 
            icon={<ChevronRight className="rotate-90 w-4 h-4 text-gray-400" />}
            color="gray"
          />
          <PileSection 
            title="Exhaust Pile" 
            cards={state.exhaustPile} 
            icon={<Trash2 className="w-4 h-4 text-gray-400" />}
            color="gray"
          />
        </aside>

        {/* Center/Right: Action Probabilities & Charts */}
        <div className="col-span-9 space-y-6">
          <div className="grid grid-cols-2 gap-6">
            {/* Enemy List */}
            <section className="glass p-6">
              <h3 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-4 flex items-center gap-2">
                <Sword className="w-4 h-4" /> Enemy Status
              </h3>
              <div className="space-y-4">
                {state.enemies.map((enemy, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium">{enemy.name}</span>
                      <span className="opacity-70">{enemy.hp} / {enemy.maxHp} HP</span>
                    </div>
                    <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${(enemy.hp / enemy.maxHp) * 100}%` }}
                        transition={{ duration: 0.1 }}
                        className="h-full bg-red-600 shadow-[0_0_8px_rgba(220,38,38,0.5)]"
                      />
                    </div>
                    {enemy.block > 0 && (
                      <div className="text-xs text-blue-400 flex items-center gap-1">
                        <Shield className="w-3 h-3" /> {enemy.block} Block
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* V-Value Graph */}
            <section className="glass p-6 min-h-[300px]">
               <h3 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" /> V-Value Trend
              </h3>
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={vHistory}>
                    <Line 
                      type="monotone" 
                      dataKey="v" 
                      stroke="#00ffcc" 
                      strokeWidth={2} 
                      dot={false}
                      animationDuration={50}
                    />
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e202d', border: 'none', borderRadius: '8px' }}
                      itemStyle={{ color: '#00ffcc' }}
                    />
                    {events.map((ev, idx) => (
                      <ReferenceLine 
                        key={idx} 
                        x={ev.time} 
                        stroke={ev.color} 
                        strokeDasharray="4 4"
                        strokeWidth={1}
                        ifOverflow="visible"
                      >
                        <Label 
                          value={ev.label} 
                          position="top" 
                          fill={ev.color} 
                          fontSize={9} 
                          fontWeight="bold"
                          offset={10}
                        />
                      </ReferenceLine>
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>

          {/* Action Probabilities */}
          <section className="glass p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-6 flex items-center gap-2">
              <Activity className="w-4 h-4" /> AI Action Probabilities (Top 10)
            </h3>
            <div className="space-y-5">
              <AnimatePresence mode="popLayout">
                {top_actions.map((action) => (
                  <motion.div 
                    layout
                    key={action.id} 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.1 }}
                    className="group"
                  >
                    <div className="flex justify-between items-end mb-1">
                      <span className="text-sm font-medium group-hover:text-brand-secondary transition-colors truncate max-w-[80%]">
                        {action.name}
                      </span>
                      <span className="text-xs font-mono text-brand-secondary opacity-80">
                        {(action.prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${action.prob * 100}%` }}
                        transition={{ duration: 0.1 }}
                        className="h-full bg-brand-secondary shadow-[0_0_10px_rgba(0,255,204,0.4)]"
                      />
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </section>
        </div>
      </main>
      
      {!connected && (
         <div className="fixed bottom-4 right-4 bg-red-900/80 text-white text-xs px-3 py-1.5 rounded-full border border-red-500 backdrop-blur-sm animate-pulse">
            Connection Lost - Reconnecting...
         </div>
      )}
    </div>
  );
};

const StatCard = ({ label, value, icon, percent }: any) => {
  return (
    <div className="glass p-4 relative overflow-hidden group">
      <div className="flex items-center gap-3 mb-1">
        <div className="p-2 bg-white/5 rounded-lg group-hover:bg-white/10 transition-colors">
          {icon}
        </div>
        <span className="text-[10px] font-bold uppercase tracking-widest text-gray-500">{label}</span>
      </div>
      <div className="text-2xl font-mono tracking-tight font-semibold">
        {value}
      </div>
      {percent !== undefined && (
        <div className="absolute bottom-0 left-0 h-[2px] bg-red-600 transition-all duration-100" style={{ width: `${percent}%` }} />
      )}
    </div>
  );
};

const PileSection = ({ title, cards, icon, current }: any) => {
  const [collapsed, setCollapsed] = useState(!current && cards.length > 5);
  
  // If it's a list of strings, convert to objects
  const processedCards = cards.map((c: any) => typeof c === 'string' ? { name: c, id: c } : c);

  return (
    <div className={cn(
      "glass overflow-hidden flex flex-col transition-all duration-300",
      current ? "max-h-[400px] border-brand-secondary/20 shadow-[0_0_20px_rgba(0,255,204,0.05)]" : "max-h-[160px] opacity-80 hover:opacity-100"
    )}>
      <div className="p-3 border-b border-white/5 flex justify-between items-center bg-white/5">
        <h4 className="text-xs font-bold uppercase tracking-wider flex items-center gap-2">
          {icon} {title} ({cards.length})
        </h4>
        {cards.length > 5 && (
          <button onClick={() => setCollapsed(!collapsed)} className="text-[10px] text-gray-500 hover:text-white uppercase tracking-tighter">
            {collapsed ? 'View All' : 'Collapse'}
          </button>
        )}
      </div>
      <div className={cn(
        "p-2 space-y-1.5 overflow-y-auto custom-scrollbar",
        collapsed ? "max-h-24" : "flex-1"
      )}>
        {processedCards.length === 0 ? (
          <div className="text-[10px] text-gray-600 italic py-2 text-center">Empty</div>
        ) : (
          processedCards.map((card: any, idx: number) => (
            <div key={`${card.id}-${idx}`} className={cn(
              "text-[11px] px-2 py-1.5 rounded bg-white/5 border border-white/5 flex justify-between group hover:bg-white/10 transition-colors",
              card.is_generated && "border-brand-secondary/30 bg-brand-secondary/5"
            )}>
              <span className="font-medium truncate pr-2">{card.name}</span>
              <span className="opacity-40 font-mono text-[9px] shrink-0">{card.cost}E</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default App;
