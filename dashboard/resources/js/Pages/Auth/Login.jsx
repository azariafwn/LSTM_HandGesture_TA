import Checkbox from '@/Components/Checkbox';
import InputError from '@/Components/InputError';
import InputLabel from '@/Components/InputLabel';
import PrimaryButton from '@/Components/PrimaryButton';
import TextInput from '@/Components/TextInput';
import { Head, Link, useForm } from '@inertiajs/react';

export default function Login({ status, canResetPassword }) {
    const { data, setData, post, processing, errors, reset } = useForm({
        email: '',
        password: '',
        remember: false,
    });

    const submit = (e) => {
        e.preventDefault();
        post(route('login'), {
            onFinish: () => reset('password'),
        });
    };

    return (
        // Wrapper Utama (Gaya Cyberpunk/Futuristik)
        <div className="min-h-screen flex flex-col sm:justify-center items-center pt-6 sm:pt-0 bg-slate-950 font-mono text-slate-300">
            <Head title="Log in" />

            {/* Header / Judul */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                    SYSTEM_ACCESS
                </h1>
                <p className="text-xs text-slate-500 mt-1 tracking-widest">SECURE LOGIN REQUIRED</p>
            </div>

            {/* Container Form */}
            <div className="w-full sm:max-w-md mt-6 px-6 py-8 bg-slate-900/50 border border-slate-800 shadow-[0_0_15px_rgba(0,0,0,0.5)] overflow-hidden sm:rounded-xl backdrop-blur-sm relative group">
                
                {/* Efek Sudut Dekoratif */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/50 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/50 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/50 rounded-br-lg"></div>

                {status && (
                    <div className="mb-4 text-sm font-medium text-emerald-400 bg-emerald-900/20 p-3 rounded border border-emerald-800">
                        {status}
                    </div>
                )}

                <form onSubmit={submit}>
                    <div>
                        <InputLabel htmlFor="email" value="IDENTITY_KEY (Email)" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="email"
                            type="email"
                            name="email"
                            value={data.email}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="username"
                            isFocused={true}
                            onChange={(e) => setData('email', e.target.value)}
                            placeholder="user@system.local"
                        />

                        <InputError message={errors.email} className="mt-2 text-rose-400" />
                    </div>

                    <div className="mt-4">
                        <InputLabel htmlFor="password" value="ACCESS_CODE (Password)" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="password"
                            type="password"
                            name="password"
                            value={data.password}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="current-password"
                            onChange={(e) => setData('password', e.target.value)}
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password} className="mt-2 text-rose-400" />
                    </div>

                    <div className="block mt-4">
                        <label className="flex items-center">
                            <Checkbox
                                name="remember"
                                checked={data.remember}
                                onChange={(e) => setData('remember', e.target.checked)}
                                className="bg-slate-950 border-slate-700 text-cyan-600 focus:ring-cyan-500 rounded"
                            />
                            <span className="ms-2 text-sm text-slate-400 font-mono">REMEMBER_SESSION</span>
                        </label>
                    </div>

                    <div className="flex items-center justify-between mt-6">
                        {canResetPassword && (
                            <Link
                                href={route('password.request')}
                                className="underline text-xs text-slate-500 hover:text-cyan-400 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors"
                            >
                                Forgot Code?
                            </Link>
                        )}

                        <PrimaryButton 
                            className="ms-4 bg-cyan-600 hover:bg-cyan-500 text-white font-bold tracking-wider px-6 py-2 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] transition-all border border-cyan-400" 
                            disabled={processing}
                        >
                            {processing ? 'AUTHENTICATING...' : 'INITIALIZE'}
                        </PrimaryButton>
                    </div>

                    {/* --- BAGIAN BARU: LINK REGISTER --- */}
                    <div className="mt-8 pt-4 border-t border-slate-800 text-center">
                        <span className="text-xs text-slate-500 tracking-wider">NO CREDENTIALS?</span>
                        <Link 
                            href={route('register')}
                            className="ml-2 text-xs text-cyan-400 hover:text-cyan-300 font-bold tracking-widest hover:underline hover:shadow-[0_0_8px_rgba(34,211,238,0.4)] transition-all"
                        >
                            REGISTER_NEW_AGENT
                        </Link>
                    </div>
                </form>
            </div>
            
            {/* Footer Style */}
            <div className="mt-8 text-[10px] text-slate-600 tracking-widest">
                SECURE SYSTEM V1.0 • UNAUTHORIZED ACCESS PROHIBITED
            </div>
        </div>
    );
}