import InputError from '@/Components/InputError';
import InputLabel from '@/Components/InputLabel';
import PrimaryButton from '@/Components/PrimaryButton';
import TextInput from '@/Components/TextInput';
import { Head, Link, useForm } from '@inertiajs/react';

export default function Register() {
    const { data, setData, post, processing, errors, reset } = useForm({
        name: '',
        email: '',
        password: '',
        password_confirmation: '',
    });

    const submit = (e) => {
        e.preventDefault();

        post(route('register'), {
            onFinish: () => reset('password', 'password_confirmation'),
        });
    };

    return (
        // Wrapper Utama (Gaya Cyberpunk/Futuristik)
        <div className="min-h-screen flex flex-col sm:justify-center items-center pt-6 sm:pt-0 bg-slate-950 font-mono text-slate-300">
            <Head title="Register" />

            {/* Header / Judul */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                    NEW_USER_ENTRY
                </h1>
                <p className="text-xs text-slate-500 mt-1 tracking-widest">CREATE SECURE IDENTITY</p>
            </div>

            {/* Container Form */}
            <div className="w-full sm:max-w-md mt-6 px-6 py-8 bg-slate-900/50 border border-slate-800 shadow-[0_0_15px_rgba(0,0,0,0.5)] overflow-hidden sm:rounded-xl backdrop-blur-sm relative group">
                
                {/* Efek Sudut Dekoratif */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/50 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/50 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/50 rounded-br-lg"></div>

                <form onSubmit={submit}>
                    {/* Name Input */}
                    <div>
                        <InputLabel htmlFor="name" value="AGENT_NAME" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="name"
                            name="name"
                            value={data.name}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="name"
                            isFocused={true}
                            onChange={(e) => setData('name', e.target.value)}
                            required
                            placeholder="John Doe"
                        />

                        <InputError message={errors.name} className="mt-2 text-rose-400" />
                    </div>

                    {/* Email Input */}
                    <div className="mt-4">
                        <InputLabel htmlFor="email" value="IDENTITY_KEY (Email)" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="email"
                            type="email"
                            name="email"
                            value={data.email}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="username"
                            onChange={(e) => setData('email', e.target.value)}
                            required
                            placeholder="user@system.local"
                        />

                        <InputError message={errors.email} className="mt-2 text-rose-400" />
                    </div>

                    {/* Password Input */}
                    <div className="mt-4">
                        <InputLabel htmlFor="password" value="SECRET_CODE" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="password"
                            type="password"
                            name="password"
                            value={data.password}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="new-password"
                            onChange={(e) => setData('password', e.target.value)}
                            required
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password} className="mt-2 text-rose-400" />
                    </div>

                    {/* Confirm Password Input */}
                    <div className="mt-4">
                        <InputLabel htmlFor="password_confirmation" value="CONFIRM_CODE" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="password_confirmation"
                            type="password"
                            name="password_confirmation"
                            value={data.password_confirmation}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="new-password"
                            onChange={(e) => setData('password_confirmation', e.target.value)}
                            required
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password_confirmation} className="mt-2 text-rose-400" />
                    </div>

                    {/* Footer Actions */}
                    <div className="flex items-center justify-between mt-6">
                        <Link
                            href={route('login')}
                            className="underline text-xs text-slate-500 hover:text-cyan-400 rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-colors"
                        >
                            Already registered?
                        </Link>

                        <PrimaryButton 
                            className="ms-4 bg-cyan-600 hover:bg-cyan-500 text-white font-bold tracking-wider px-6 py-2 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] transition-all border border-cyan-400" 
                            disabled={processing}
                        >
                            REGISTER
                        </PrimaryButton>
                    </div>
                </form>
            </div>

            {/* Footer Style */}
            <div className="mt-8 text-[10px] text-slate-600 tracking-widest">
                SECURE SYSTEM V1.0 • NEW PERSONNEL REGISTRATION
            </div>
        </div>
    );
}