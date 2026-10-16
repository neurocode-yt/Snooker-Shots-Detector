/* Source families and independent highlight versions; no media/detection work. */
(function (root) {
  const UI = root.snookerUI || (typeof module !== 'undefined' ? require('./ui.js') : null);
  const types = Object.freeze([
    {id:'all', label:'All shots'},
    {id:'pots_and_fouls', label:'Pots + fouls'},
    {id:'pots_only', label:'Pots only'},
  ]);
  function sourcePath(job) {
    let path = String(job.source_fingerprint?.path || job.source_path || `job:${job.job_id}`).replace(/\\/g,'/');
    if (/^[a-z]:\//i.test(path) || path.startsWith('//')) path = path.toLowerCase();
    return `${job.algorithm || 'classic'}:${path}`;
  }
  function revision(job) {
    const fp = job.source_fingerprint;
    return fp && fp.size != null && fp.mtime_ns != null ? `${fp.size}:${fp.mtime_ns}` : null;
  }
  const created = job => UI.finished(job) || Boolean(job.has_saved_detection);
  function rank(job) {
    return UI.active(job) ? 4 : job.download_url ? 3 : UI.finished(job) ? 2 : 1;
  }
  function groups(jobs) {
    const buckets = new Map(), families = new Map();
    for (const job of jobs) {
      const path = sourcePath(job);
      if (!buckets.has(path)) buckets.set(path,new Set());
      if (revision(job)) buckets.get(path).add(revision(job));
    }
    for (const job of jobs) {
      const path = sourcePath(job), known = buckets.get(path);
      // Legacy jobs lack fingerprints. Associate them only when the source has
      // one known revision; never bridge two different replacements of a file.
      const rev = revision(job) || (known.size === 1 ? [...known][0] : known.size ? `unverified:${job.job_id}` : 'legacy');
      const key = JSON.stringify([path,rev]);
      if (!families.has(key)) families.set(key,{key,jobs:[]});
      families.get(key).jobs.push(job);
    }
    return [...families.values()].map(group => {
      group.jobs.sort((a,b)=>Number(b.created_at||0)-Number(a.created_at||0));
      group.titleJob = group.jobs.find(job=>(job.selection||'all')==='all') || group.jobs[0];
      group.updated = Math.max(...group.jobs.map(job=>Number(job.created_at||0)));
      group.versions = types.map(type=>{
        const runs = group.jobs.filter(job=>(job.selection||'all')===type.id)
          .sort((a,b)=>rank(b)-rank(a)||Number(b.created_at||0)-Number(a.created_at||0));
        return {...type,job:runs[0]||null,history:runs.slice(1)};
      });
      group.base = group.jobs.filter(job=>!UI.active(job) && UI.finished(job)
        && (job.can_create_variants || job.can_create_points))
        .sort((a,b)=>Number((b.selection||'all')==='all')-Number((a.selection||'all')==='all')
          || Number(b.created_at||0)-Number(a.created_at||0))[0] || null;
      group.created = group.versions.filter(version=>[version.job,...version.history].some(job=>job&&created(job))).length;
      group.missing = group.versions.filter(version=>!version.job).length;
      return group;
    }).sort((a,b)=>b.updated-a.updated);
  }
  function matches(group, filter, query='') {
    if (filter==='missing') return group.missing>0 && group.jobs.some(job=>UI.matches(job,'all',query));
    return group.jobs.some(job=>UI.matches(job,filter,query));
  }
  function state(job) {
    if (UI.finished(job) && !UI.active(job) && (job.pot_selection_summary?.score_readings===0
        || job.pot_selection_summary?.included===0 && Number(job.pot_selection_summary?.unknown||0)>0)) {
      return {label:'Score check needed',tone:'warning'};
    }
    if (UI.finished(job) && !UI.active(job) && !job.download_url && !job.export_needs_update) {
      return {label:'Shots ready',tone:'warning'};
    }
    return UI.status(job);
  }
  const api = Object.freeze({types,groups,matches,state,sourcePath});
  root.snookerLibrary = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
