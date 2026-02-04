import AuthenticatedLayout from '@/Layouts/AuthenticatedLayout';
import { Head, router } from '@inertiajs/react';
import { useState, useEffect } from 'react';

export default function Dashboard({ auth, logs }) {
    const [streamUrl, setStreamUrl] = useState('');
    const [isSyncing, setIsSyncing] = useState(false);
    const [darkMode, setDarkMode] = useState(true);

    const formatTime = (timestamp) => {
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('id-ID', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                hour12: false 
            });
        } catch (e) {
            return "-";
        }
    };

    useEffect(() => {
        setStreamUrl(`http://${window.location.hostname}:5000/video_feed`);
        const interval = setInterval(() => {
            setIsSyncing(true);
            router.reload({
                only: ['logs'],
                preserveScroll: true,
                preserveState: true,
                onFinish: () => setIsSyncing(false),
            });
        }, 200); 
        return () => clearInterval(interval);
    }, []);

    const initialStatus = { D1: 'OFF', D2: 'OFF', D3: 'OFF', D4: 'OFF' };
    const getDeviceStatuses = () => {
        const currentStatus = { ...initialStatus };
        const found = { D1: false, D2: false, D3: false, D4: false };
        logs.forEach(log => {
            const cmd = log.last_command;
            if (!found.D1 && cmd.includes('D1')) { currentStatus.D1 = cmd.includes('ON') ? 'ON' : 'OFF'; found.D1 = true; }
            if (!found.D2 && cmd.includes('D2')) { currentStatus.D2 = cmd.includes('ON') ? 'ON' : 'OFF'; found.D2 = true; }
            if (!found.D3 && cmd.includes('D3')) { currentStatus.D3 = cmd.includes('ON') ? 'ON' : 'OFF'; found.D3 = true; }
            if (!found.D4 && cmd.includes('D4')) { currentStatus.D4 = cmd.includes('ON') ? 'ON' : 'OFF'; found.D4 = true; }
        });
        return currentStatus;
    };

    const deviceStatus = getDeviceStatuses();

    const theme = {
        bg: darkMode ? 'bg-slate-950' : 'bg-gray-100',
        text: darkMode ? 'text-slate-300' : 'text-gray-800',
        cardBg: darkMode ? 'bg-slate-900/50' : 'bg-white',
        cardBorder: darkMode ? 'border-slate-800' : 'border-gray-200',
        tableHeader: darkMode ? 'bg-slate-900/80 text-slate-500' : 'bg-gray-200 text-gray-600',
        tableRowHover: darkMode ? 'hover:bg-cyan-900/10' : 'hover:bg-blue-50',
        tableText: darkMode ? 'text-slate-400' : 'text-gray-700',
        accentText: darkMode ? 'text-cyan-500' : 'text-blue-600',
        headerBorder: darkMode ? 'border-slate-800/50' : 'border-gray-300'
    };

    const DeviceCard = ({ name, status }) => (
        <div className={`relative group p-4 sm:p-5 rounded-xl border transition-all duration-300 
            ${status === 'ON' 
                ? (darkMode ? 'bg-cyan-950/30 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]' : 'bg-blue-100 border-blue-400 shadow-lg')
                : `${theme.cardBg} ${theme.cardBorder}`
            } flex flex-col items-center justify-center backdrop-blur-sm`}>
            
            <div className={`absolute top-3 right-3 w-2 sm:w-3 h-2 sm:h-3 rounded-full ${status === 'ON' ? 'bg-cyan-400 animate-pulse' : 'bg-slate-500'}`}></div>

            <div className={`font-mono text-xs sm:text-sm tracking-widest mb-1 sm:mb-2 font-bold ${status === 'ON' ? (darkMode ? 'text-cyan-100' : 'text-blue-900') : (darkMode ? 'text-slate-500' : 'text-gray-500')}`}>
                {name.toUpperCase()}
            </div>
            
            <div className={`text-xl sm:text-2xl font-bold font-mono tracking-tighter ${status === 'ON' ? (darkMode ? 'text-cyan-400' : 'text-blue-600') : (darkMode ? 'text-slate-600' : 'text-gray-400')}`}>
                {status === 'ON' ? 'ONLINE' : 'OFFLINE'}
            </div>
        </div>
    );

    return (
        <AuthenticatedLayout user={auth.user} header={null}>
            <Head title="Dashboard" />

            <div className={`min-h-screen font-mono selection:bg-cyan-500 selection:text-black transition-colors duration-500 ${theme.bg} ${theme.text}`}>
                
                {/* Custom Header: Responsive Flex-Col di Mobile */}
                <div className={`max-w-7xl mx-auto px-4 sm:px-6 pt-6 sm:pt-10 pb-4 sm:pb-6 flex flex-col sm:flex-row justify-between items-start sm:items-end border-b ${theme.headerBorder} mb-6 sm:mb-8 gap-4 sm:gap-0`}>
                    <div>
                        <h1 className="text-2xl sm:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                            SYSTEM_DASHBOARD
                        </h1>
                        <p className={`text-xs sm:text-sm mt-1 tracking-widest ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>SECURE CONNECTION ESTABLISHED</p>
                    </div>
                    <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
                         {/* Tombol Dark Mode */}
                         <button 
                            onClick={() => setDarkMode(!darkMode)}
                            className={`px-3 py-2 sm:px-4 sm:py-2 rounded-lg border transition text-[10px] sm:text-xs font-bold tracking-wider w-full sm:w-auto text-center
                                ${darkMode 
                                    ? 'border-slate-700 hover:border-cyan-500 text-slate-400 hover:text-cyan-400' 
                                    : 'border-gray-300 hover:border-blue-500 text-gray-600 hover:text-blue-600 bg-white shadow-sm'}`}
                        >
                            {darkMode ? '☀ LIGHT_MODE' : '☾ DARK_MODE'}
                        </button>
                    </div>
                </div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
                    
                    {/* Grid Layout: Stack di Mobile (grid-cols-1) */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                        {/* 1. LAYAR KAMERA */}
                        <div className="lg:col-span-1 relative group w-full">
                            {darkMode && (
                                <>
                                    <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500 z-20"></div>
                                    <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500 z-20"></div>
                                    <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500 z-20"></div>
                                    <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500 z-20"></div>
                                </>
                            )}

                            <div className={`bg-black rounded-lg overflow-hidden shadow-2xl border relative h-56 sm:h-64 lg:h-full min-h-[220px] sm:min-h-[250px] flex items-center justify-center ${theme.cardBorder}`}>
                                {darkMode && <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 pointer-events-none bg-[length:100%_4px,3px_100%]"></div>}
                                
                                <div className="absolute top-3 left-3 sm:top-4 sm:left-4 flex items-center gap-2 z-20">
                                    <span className="relative flex h-3 w-3">
                                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                                    </span>
                                    <span className="text-[10px] font-bold text-red-500 tracking-widest bg-black/50 px-2 py-0.5 rounded">REC ● LIVE</span>
                                </div>

                                <img 
                                    src={streamUrl} 
                                    alt="FEED_LOST" 
                                    className="h-full w-full object-contain bg-slate-900 opacity-90"
                                    onError={(e) => { e.target.style.display = 'none'; }}
                                />
                                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-700 -z-0">
                                    <span className="text-sm tracking-widest font-bold">SIGNAL_LOST</span>
                                </div>
                            </div>
                        </div>

                        {/* 2. PANEL STATUS */}
                        <div className="lg:col-span-2 flex flex-col gap-4">
                            <div className={`${theme.cardBg} backdrop-blur border ${theme.cardBorder} p-4 sm:p-5 rounded-xl flex justify-between items-center shadow-sm`}>
                                <div>
                                    <h3 className={`font-bold text-sm sm:text-base tracking-widest ${theme.accentText}`}>DEVICE_STATUS</h3>
                                    <p className={`text-[10px] sm:text-xs uppercase mt-1 ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>Monitoring Node: Active</p>
                                </div>
                                <div className={`flex items-center gap-2 border px-2 sm:px-3 py-1 sm:py-1.5 rounded text-[10px] sm:text-xs transition-all ${darkMode ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-gray-50 border-gray-200 text-gray-600'}`}>
                                    <div className={`w-1.5 sm:w-2 h-1.5 sm:h-2 rounded-full ${isSyncing ? 'bg-cyan-400 animate-ping' : 'bg-slate-500'}`}></div>
                                    <span className="font-mono font-bold">{isSyncing ? 'SYNCING...' : 'IDLE'}</span>
                                </div>
                            </div>

                            {/* Responsive Grid untuk Kartu: 2 kolom di mobile, 4 kolom di desktop */}
                            <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 h-full">
                                <DeviceCard name="Light_01" status={deviceStatus.D1} />
                                <DeviceCard name="Light_02" status={deviceStatus.D2} />
                                <DeviceCard name="Fan_SYS" status={deviceStatus.D3} />
                                <DeviceCard name="Light_04" status={deviceStatus.D4} />
                            </div>
                        </div>
                    </div>

                    {/* 3. TABEL LOG */}
                    <div className={`border ${theme.cardBorder} ${darkMode ? 'bg-black/40' : 'bg-white'} rounded-xl overflow-hidden backdrop-blur-sm shadow-sm`}>
                        <div className={`p-4 border-b ${theme.cardBorder} ${darkMode ? 'bg-slate-900/30' : 'bg-gray-50'} flex justify-between items-center`}>
                            <h3 className={`text-sm sm:text-base font-bold tracking-wider ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>Activity_Logs</h3>
                            <span className={`text-[10px] sm:text-xs font-mono ${darkMode ? 'text-slate-600' : 'text-gray-400'}`}>Auto-refresh 200ms</span>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="min-w-full text-xs sm:text-sm font-mono">
                                <thead className={theme.tableHeader}>
                                    <tr>
                                        <th className="py-3 sm:py-4 px-4 sm:px-6 text-left font-bold tracking-wider">TIME</th>
                                        <th className="py-3 sm:py-4 px-4 sm:px-6 text-left font-bold tracking-wider">GESTURE</th>
                                        <th className="py-3 sm:py-4 px-4 sm:px-6 text-left font-bold tracking-wider">CMD</th>
                                        <th className="py-3 sm:py-4 px-4 sm:px-6 text-center font-bold tracking-wider hidden sm:table-cell">LATENCY</th>
                                        <th className="py-3 sm:py-4 px-4 sm:px-6 text-center font-bold tracking-wider">STATE</th>
                                    </tr>
                                </thead>
                                <tbody className={`divide-y ${darkMode ? 'divide-slate-800/50' : 'divide-gray-200'}`}>
                                    {logs.map((log, index) => (
                                        <tr key={index} className={`transition-colors duration-200 ${theme.tableRowHover} ${index === 0 && darkMode ? 'bg-cyan-950/20' : ''} ${index === 0 && !darkMode ? 'bg-blue-50' : ''}`}>
                                            <td className={`py-3 sm:py-4 px-4 sm:px-6 ${theme.tableText}`}>{formatTime(log.timestamp)}</td>
                                            <td className={`py-3 sm:py-4 px-4 sm:px-6 font-bold ${darkMode ? 'text-cyan-300' : 'text-blue-700'}`}>{log.target_gesture}</td>
                                            <td className="py-3 sm:py-4 px-4 sm:px-6">
                                                <span className={`inline-block px-2 py-1 rounded text-[10px] sm:text-xs border font-bold ${
                                                    log.last_command.includes('ON') 
                                                    ? (darkMode ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10' : 'border-green-300 text-green-700 bg-green-100')
                                                    : (darkMode ? 'border-rose-500/30 text-rose-400 bg-rose-500/10' : 'border-red-300 text-red-700 bg-red-100')
                                                }`}>
                                                    {log.last_command}
                                                </span>
                                            </td>
                                            {/* Latency disembunyikan di mobile agar tabel muat */}
                                            <td className={`py-3 sm:py-4 px-4 sm:px-6 text-center hidden sm:table-cell ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>{parseFloat(log.wifi_latency_ms).toFixed(0)}ms</td>
                                            <td className="py-3 sm:py-4 px-4 sm:px-6 text-center">
                                                {log.last_command.includes('ON') 
                                                    ? <span className="text-emerald-400 text-lg">●</span> 
                                                    : <span className="text-slate-400 text-lg">○</span>
                                                }
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                </div>
            </div>
        </AuthenticatedLayout>
    );
}