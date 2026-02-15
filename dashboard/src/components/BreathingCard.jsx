import React from 'react';
import { Wind, Activity, Timer, AlertTriangle } from 'lucide-react';

const BreathingCard = ({ data }) => {
    if (!data) return null;
    const { breath_rate_bpm, interval_cv, num_breath_events, irregular, laboured } = data;

    return (
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                    <Wind className="w-6 h-6 text-blue-500" />
                    Breathing Analysis
                </h3>
                {(irregular || laboured) && <AlertTriangle className="w-6 h-6 text-orange-500" />}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-blue-50 p-4 rounded-2xl">
                    <div className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-1">BPM</div>
                    <div className="text-3xl font-bold text-gray-800">{breath_rate_bpm}</div>
                    <div className="text-xs text-blue-400 mt-1">Breaths per minute</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-2xl">
                    <div className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-1">Events</div>
                    <div className="text-3xl font-bold text-gray-800">{num_breath_events}</div>
                    <div className="text-xs text-purple-400 mt-1">Detected breaths</div>
                </div>
            </div>

            <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                    <div className="flex items-center gap-3">
                        <Timer className="w-5 h-5 text-gray-400" />
                        <span className="text-gray-600">Interval Consistency (CV)</span>
                    </div>
                    <span className="font-semibold text-gray-800">{interval_cv !== null ? interval_cv : 'N/A'}</span>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                    <div className="flex items-center gap-3">
                        <Activity className="w-5 h-5 text-gray-400" />
                        <span className="text-gray-600">Breathing Pattern</span>
                    </div>
                    <span className={`font-semibold ${irregular ? 'text-orange-500' : 'text-green-500'}`}>
                        {irregular ? 'Irregular' : 'Regular'}
                    </span>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                    <div className="flex items-center gap-3">
                        <Wind className="w-5 h-5 text-gray-400" />
                        <span className="text-gray-600">Effort</span>
                    </div>
                    <span className={`font-semibold ${laboured ? 'text-red-500' : 'text-green-500'}`}>
                        {laboured ? 'Laboured' : 'Normal'}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default BreathingCard;
