-- 008_waitlist.sql
--
-- Stores landing-page waitlist signups. Single opt-in: the row is
-- created on submit and a "you're on the list" email is sent right
-- away. citext gives case-insensitive uniqueness so Foo@x.com and
-- foo@x.com collide. RLS is enabled with no policies — the API route
-- writes via the service role (which bypasses RLS); anon/authenticated
-- clients have no access.

create extension if not exists citext;

create table if not exists waitlist (
  id          uuid primary key default gen_random_uuid(),
  email       citext not null unique,
  source      text not null default 'landing-hero',
  user_agent  text,
  created_at  timestamptz not null default now()
);

create index if not exists waitlist_created_idx on waitlist (created_at desc);

alter table waitlist enable row level security;
