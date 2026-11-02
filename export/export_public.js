// Read-only export for the public dataset. Player ids are random per run; the mapping is never written anywhere.
// Only fields needed by the experiments are kept: order, season, mode, teams, winner (no times, maps, scores, names).
const D = require('better-sqlite3'); const c = require('crypto');
const db = new D('/data/match_history.db', { readonly: true, fileMustExist: true });
const ids = new Map(); const anon = s => { if (!ids.has(s)) ids.set(s, 'p' + c.randomBytes(5).toString('hex')); return ids.get(s); };
const ms = db.prepare("SELECT match_id, season_id, game_mode, winner FROM matches WHERE winner IN (1,2) ORDER BY finished_at, id").all();
const ps = db.prepare("SELECT match_id, steam_id, team FROM match_players").all();
const by = {}; for (const p of ps) (by[p.match_id] = by[p.match_id] || []).push(p);
// shuffle player-id assignment order so ids carry no information about first appearance
const all = [...new Set(ps.map(p => p.steam_id))]; for (let i = all.length - 1; i > 0; i--) { const j = c.randomInt(i + 1); [all[i], all[j]] = [all[j], all[i]]; } all.forEach(anon);
const out = []; let k = 0;
for (const m of ms) {
  const pl = by[m.match_id] || []; const t1 = pl.filter(p => p.team === 1).map(p => anon(p.steam_id)).sort(); const t2 = pl.filter(p => p.team === 2).map(p => anon(p.steam_id)).sort();
  if (!t1.length || !t2.length) continue;
  out.push({ order: k++, season: m.season_id, mode: m.game_mode, winner: m.winner, team1: t1, team2: t2 });
}
process.stdout.write(JSON.stringify(out));
