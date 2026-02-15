import React from 'react';
import { Brain, Activity, AlertCircle, CheckCircle } from 'lucide-react';

const DementiaCard = ({ data }) => {
    if (!data) return null;
    const { label, score, confidence, explanation } = data;

    const isHealthy = label === 'Healthy';
    const scoreColor = isHealthy ? 'text-green-600' : 'text-red-500';
    const icon = isHealthy ? <CheckCircle className="w-6 h-6 text-green-500" /> : <AlertCircle className="w-6 h-6 text-red-500" />;

    // Normalize score for display (assuming 0-100)
    const displayScore = score !== undefined ? score : 'N/A';

    return (
        <div className="glass-card p-6 rounded-[2rem]">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                    <div className="p-2 bg-purple-100 rounded-xl">
                        <Brain className="w-5 h-5 text-purple-600" />
                    </div>
                    Dementia Evaluation
                </h3>
                {icon}
            </div>

            <div className="flex items-end gap-2 mb-6">
                <span className={`text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r ${isHealthy ? 'from-green-600 to-emerald-600' : 'from-red-500 to-pink-600'}`}>
                    {displayScore}
                </span>
                <span className="text-gray-400 text-lg mb-1 font-medium">/100</span>
            </div>

            <div className="mb-6">
                <div className="flex justify-between text-sm text-gray-500 mb-2 font-medium">
                    <span>Confidence Score</span>
                    <span>{confidence}</span>
                </div>
                <div className="w-full bg-gray-200/50 rounded-full h-2">
                    <div
                        className={`h-2 rounded-full shadow-sm ${isHealthy ? 'bg-gradient-to-r from-green-400 to-emerald-500' : 'bg-gradient-to-r from-red-500 to-pink-600'}`}
                        style={{ width: confidence === 'High' ? '90%' : confidence === 'Medium' ? '60%' : '30%' }}
                    ></div>
                </div>
            </div>

            <div className="p-4 bg-white/40 rounded-2xl border border-white/40 backdrop-blur-sm">
                <p className="text-gray-600 text-sm leading-relaxed">
                    {explanation}
                </p>
            </div>

            <div className="mt-4 flex items-center gap-2">
                <span className={`px-3 py-1 rounded-full text-sm font-bold border ${isHealthy ? 'bg-green-50/50 text-green-700 border-green-200' : 'bg-red-50/50 text-red-700 border-red-200'}`}>
                    {label}
                </span>
            </div>
        </div>
    );
};

export default DementiaCard;
