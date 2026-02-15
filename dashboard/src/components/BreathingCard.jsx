import React from 'react';
import { Wind, Activity, Timer, AlertTriangle } from 'lucide-react';

const BreathingCard = ({ data }) => {
    if (!data) return null;
    const { breath_rate_bpm, interval_cv, num_breath_events, irregular, laboured } = data;

    return (
        <div className="glass-card p-6 rounded-[2rem]">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                    <div className="p-2 bg-blue-100 rounded-xl">
                        <Wind className="w-5 h-5 text-blue-500" />
                    </div>
                    Breathing Analysis
                </h3>
                {(irregular || laboured) && <AlertTriangle className="w-5 h-5 text-orange-500" />}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-blue-50/50 p-4 rounded-2xl border border-blue-100 backdrop-blur-sm">
                    <div className="text-blue-400 text-[10px] font-bold uppercase tracking-wider mb-1">BPM</div>
                    <div className="text-3xl font-bold text-gray-800">{breath_rate_bpm}</div>
                    <div className="text-xs text-blue-400 mt-1 font-medium">Breaths / min</div>
                </div>
                <div className="bg-purple-50/50 p-4 rounded-2xl border border-purple-100 backdrop-blur-sm">
                    <div className="text-purple-400 text-[10px] font-bold uppercase tracking-wider mb-1">Events</div>
                    <div className="text-3xl font-bold text-gray-800">{num_breath_events}</div>
                    <div className="text-xs text-purple-400 mt-1 font-medium">Detected count</div>
                </div>
            </div>

            <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-white/40 rounded-xl border border-white/50">
                    <div className="flex items-center gap-3">
                        <Timer className="w-4 h-4 text-gray-400" />
                        <span className="text-gray-600 text-sm">Interval Consistency</span>
                    </div>
                    <span className="font-semibold text-gray-800 text-sm">{interval_cv !== null ? interval_cv : 'N/A'}</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-white/40 rounded-xl border border-white/50">
                    <div className="flex items-center gap-3">
                        <Activity className="w-4 h-4 text-gray-400" />
                        <span className="text-gray-600 text-sm">Breathing Pattern</span>
                    </div>
                    <span className={`font-semibold text-sm ${irregular ? 'text-orange-500' : 'text-green-500'}`}>
                        {irregular ? 'Irregular' : 'Regular'}
                    </span>
                </div>

                <div className="flex items-center justify-between p-3 bg-white/40 rounded-xl border border-white/50">
                    <div className="flex items-center gap-3">
                        <Wind className="w-4 h-4 text-gray-400" />
                        <span className="text-gray-600 text-sm">Effort</span>
                    </div>
                    <span className={`font-semibold text-sm ${laboured ? 'text-red-500' : 'text-green-500'}`}>
                        {laboured ? 'Laboured' : 'Normal'}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default BreathingCard;
