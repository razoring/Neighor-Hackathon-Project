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
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                    <Brain className="w-6 h-6 text-purple-600" />
                    Dementia Evaluation
                </h3>
                {icon}
            </div>

            <div className="flex items-end gap-2 mb-6">
                <span className={`text-5xl font-bold ${scoreColor}`}>{displayScore}</span>
                <span className="text-gray-400 text-lg mb-1">/100</span>
            </div>

            <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-500 mb-1">
                    <span>Confidence</span>
                    <span>{confidence}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                        className={`h-2.5 rounded-full ${confidence === 'High' ? 'bg-green-500' : confidence === 'Medium' ? 'bg-yellow-400' : 'bg-red-400'}`}
                        style={{ width: confidence === 'High' ? '90%' : confidence === 'Medium' ? '60%' : '30%' }}
                    ></div>
                </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-xl">
                <p className="text-gray-600 text-sm leading-relaxed">
                    {explanation}
                </p>
            </div>

            <div className="mt-4 flex items-center gap-2">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${isHealthy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    {label}
                </span>
            </div>
        </div>
    );
};

export default DementiaCard;
