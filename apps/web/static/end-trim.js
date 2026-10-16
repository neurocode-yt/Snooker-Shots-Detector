/* Saved end offsets are absolute settings, never cumulative cuts. */
(function(root) {
  function seconds(value, label) {
    const number = Number(value);
    if (String(value).trim()==='' || !Number.isFinite(number) || number<0 || number>3600)
      throw new Error(`${label} must be between 0 and 3600 seconds.`);
    return number;
  }
  function payload({normal, long, threshold, shotId, reset=false}) {
    if (reset) return {reset:true,...(shotId!=null?{shot_ids:[shotId]}:{})};
    return {seconds:seconds(normal,'End trim'), ...(shotId!=null ? {shot_ids:[shotId]} : {
      long_seconds:seconds(long,'Long-shot trim'), long_threshold:seconds(threshold,'Long-shot threshold'),
    })};
  }
  function state(shot) {
    if (!shot) return {seconds:0,description:''};
    const trim=shot.evidence?.end_trim;
    const amount=trim?.active!==false && Number.isFinite(trim?.seconds)
      ? trim.seconds : Number(shot.evidence?.end_before_ball_stop_seconds||0);
    const current=`Current clip: ${(shot.clip_end-shot.clip_start).toFixed(1)}s.`;
    const detail=trim?.active ? ` ${trim.applied_seconds.toFixed(1)}s removed from the saved ending${trim.applied_seconds+.001<trim.seconds?' (minimum playback protected)':''}.`
      : trim ? ' Previous ending restored.' : ' Save a setting, then preview before exporting.';
    return {seconds:amount,description:current+detail};
  }
  function message(data,reset=false) {
    return `${data.changed_shots} shot ending${data.changed_shots===1?'':'s'} ${reset?'restored':'changed'}.`+
      (data.limited_shots?` ${data.limited_shots} kept extra footage for minimum playback.`:'');
  }
  const api={payload,state,message};root.snookerEndTrim=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
