-- User profile metadata. Companion to Supabase's `auth.users` (managed
-- by Supabase Auth). One row per signed-up user; auto-created on first
-- backend hit via `services.persistence.profiles.get_or_create()` so we
-- don't need a database trigger touching auth.users (which requires
-- elevated privileges and complicates portable migrations).

create table if not exists profiles (
  user_id       uuid primary key references auth.users(id) on delete cascade,
  display_name  text,
  avatar_url    text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists profiles_updated_idx
  on profiles (updated_at desc);

-- RLS — users see/edit only their own profile. The backend uses the
-- service-role connection (DATABASE_URL with the postgres role), which
-- bypasses RLS, so this is defense-in-depth for any direct client
-- connections via PostgREST or future direct-from-browser queries.
alter table profiles enable row level security;

drop policy if exists "users can view their own profile" on profiles;
create policy "users can view their own profile"
  on profiles for select
  using (auth.uid() = user_id);

drop policy if exists "users can update their own profile" on profiles;
create policy "users can update their own profile"
  on profiles for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "users can insert their own profile" on profiles;
create policy "users can insert their own profile"
  on profiles for insert
  with check (auth.uid() = user_id);

-- Auto-bump updated_at on UPDATE.
create or replace function public.touch_profiles_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists profiles_updated_at on profiles;
create trigger profiles_updated_at
  before update on profiles
  for each row execute function public.touch_profiles_updated_at();
