import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, LayoutDashboard, Settings, User } from 'lucide-react';
import DementiaCard from './components/DementiaCard';
import BreathingCard from './components/BreathingCard';
import SpiderGraph from './components/SpiderGraph';

function App() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/health');
      if (response.data && (response.data.dementia_assessment || response.data.breathing)) {
        setHealthData(response.data);
      }
    } catch (error) {
      console.error("Error fetching health data", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex min-h-screen bg-[#f3f3f1]">
      {/* Sidebar */}
      <aside className="w-24 bg-white flex flex-col items-center py-8 rounded-r-3xl shadow-sm z-10 hidden md:flex">
        <div className="mb-12">
          {/* Logo placeholder */}
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-orange-400 flex items-center justify-center text-white font-bold">
            D
          </div>
        </div>
        <nav className="flex flex-col gap-8">
          <button className="p-3 bg-gray-100 rounded-xl text-gray-800 shadow-inner">
            <LayoutDashboard className="w-6 h-6" />
          </button>
          <button className="p-3 text-gray-400 hover:text-gray-600 transition-colors">
            <Activity className="w-6 h-6" />
          </button>
          <button className="p-3 text-gray-400 hover:text-gray-600 transition-colors">
            <User className="w-6 h-6" />
          </button>
        </nav>
        <div className="mt-auto">
          <button className="p-3 text-gray-400 hover:text-gray-600 transition-colors">
            <Settings className="w-6 h-6" />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto">
        <header className="flex justify-between items-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <div className="flex items-center gap-4 bg-white px-4 py-2 rounded-full shadow-sm">
            <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden">
              <User className="w-full h-full text-gray-400 p-1" />
            </div>
            <span className="text-sm font-medium text-gray-700">Patient View</span>
          </div>
        </header>

        {/* Banner / Mental Health Bar */}
        <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 mb-8 relative overflow-hidden">
          <div className="flex justify-between items-end mb-4 relative z-10">
            <div>
              <h2 className="text-xl font-semibold text-gray-800">Mental Health & Respiratory Status</h2>
              <p className="text-gray-500 mt-1">Real-time analysis from voice data</p>
            </div>

            {healthData?.dementia_assessment?.label === "Healthy" && (
              <div className="bg-green-100 text-green-700 px-4 py-1 rounded-full text-sm font-semibold">
                Status: Stable
              </div>
            )}
          </div>

          {/* Gradient Bar */}
          <div className="h-16 w-full rounded-2xl bg-gradient-to-r from-purple-400 via-yellow-300 to-red-400 relative mt-6 opacity-90">
            {/* Indicator Line */}
            <div className="absolute top-0 bottom-0 w-1 bg-white shadow-xl transform translate-x-[50%] left-[30%]" style={{ left: healthData?.dementia_assessment?.score ? `${healthData.dementia_assessment.score}%` : '50%' }}>
              <div className="w-4 h-4 bg-white rounded-full absolute -top-2 -left-1.5 shadow-md"></div>
              <div className="w-4 h-4 bg-white rounded-full absolute -bottom-2 -left-1.5 shadow-md"></div>
            </div>
          </div>
          <div className="flex justify-between mt-2 text-xs font-semibold text-gray-400 uppercase tracking-widest px-1">
            <span>Low Risk</span>
            <span>Moderate</span>
            <span>High Risk</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {healthData ? (
                <>
                  <DementiaCard data={healthData.dementia_assessment} />
                  <BreathingCard data={healthData.breathing} />
                </>
              ) : (
                <div className="col-span-2 bg-white rounded-3xl p-12 text-center text-gray-400">
                  <div className="animate-pulse flex flex-col items-center">
                    <Activity className="w-12 h-12 mb-4 text-purple-300" />
                    <p className="text-lg">Waiting for voice analysis...</p>
                    <p className="text-sm mt-2">Start a call in the backend terminal</p>
                  </div>
                </div>
              )}
            </div>

            {/* Activity List (Mock Data for Visual Completeness) */}
            <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold text-gray-800">Recent Activity</h3>
                <span className="text-sm text-gray-400 cursor-pointer">Sort v</span>
              </div>
              <div className="space-y-4">
                {[
                  { title: 'Morning Check-in', time: '10:30 AM', duration: '2 mins', icon: '☀️', color: 'bg-orange-100' },
                  { title: 'Evening Analysis', time: 'Yesterday', duration: '5 mins', icon: '🌙', color: 'bg-purple-100' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center justify-between p-4 hover:bg-gray-50 rounded-2xl transition-colors cursor-pointer group">
                    <div className="flex items-center gap-4">
                      <div className={`w-12 h-12 rounded-2xl ${item.color} flex items-center justify-center text-xl`}>
                        {item.icon}
                      </div>
                      <div>
                        <h4 className="font-semibold text-gray-800 group-hover:text-purple-600 transition-colors">{item.title}</h4>
                        <p className="text-sm text-gray-400">{item.time}</p>
                      </div>
                    </div>
                    <span className="text-sm font-medium text-gray-500">{item.duration}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Side Panel / Spider Graph */}
          <div className="lg:col-span-1 flex flex-col gap-8 h-full">
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-gray-100 text-center">
              <h3 className="text-gray-500 font-medium mb-2">Overall Health Score</h3>
              <div className="text-6xl font-bold text-gray-800 mb-1">
                {healthData?.dementia_assessment?.score ? Math.round((healthData.dementia_assessment.score + 80) / 2) : '93'}%
              </div>
              <div className="text-green-500 text-sm font-medium bg-green-50 inline-block px-3 py-1 rounded-full">
                Excellent Condition
              </div>
              <div className="mt-8">
                <SpiderGraph
                  dementiaData={healthData?.dementia_assessment}
                  breathingData={healthData?.breathing}
                />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
