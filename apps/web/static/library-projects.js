/* Library project selection, naming and resource deletion. */
(function(root) {
  const UI = root.snookerUI || (typeof module !== 'undefined' ? require('./ui.js') : null);
  const Library = root.snookerLibrary || (typeof module !== 'undefined' ? require('./library.js') : null);
  function selection(jobs, selected, filter='all', query='') {
    const all = Library.groups(jobs);
    return {all, matching:all.filter(g=>Library.matches(g,filter,query)),
      chosen:all.filter(g=>selected.has(g.key))};
  }
  function summary(plan) {
    const drafts=plan.capcut.length, uploads=plan.uploads.filter(p=>!p.endsWith('.source.json')).length;
    return `${plan.job_ids.length} saved versions and runs, including analysis, proxies, clips, MP4 exports and Shorts, plus ${drafts} linked CapCut ${drafts===1?'project':'projects'} and ${uploads} uploaded ${uploads===1?'video':'videos'} will be permanently deleted.`;
  }
  function create({getJobs, refresh, render, fetcher=(...args)=>root.fetch(...args)}) {
    const selected = new Set();
    let busy = false, currentFilter='all', currentQuery='', currentPlan=null, renameJob=null;
    const doc=root.document, el=id=>doc.getElementById(id);
    const toolbar=doc.createElement('div'); toolbar.className='project-toolbar';
    toolbar.innerHTML='<button type="button" id="projects-select-all" class="secondary">Select matching projects</button><span id="projects-selected" role="status">0 selected</span><button type="button" id="projects-clear" class="secondary" disabled>Clear selection</button><button type="button" id="projects-delete" class="danger" disabled>Delete selected</button><p id="project-cleanup" class="notice hidden">Some project resources need cleanup or recovery. <button type="button" id="projects-cleanup" class="secondary">Retry resource cleanup</button></p>';
    el('library-status').before(toolbar);
    const modal=doc.createElement('div');
    modal.innerHTML='<dialog id="project-name-dialog" class="project-dialog" aria-labelledby="project-name-title"><form id="project-name-form"><h2 id="project-name-title">Name project</h2><label>Project name<input id="project-name-input" required maxlength="120" autocomplete="off"></label><p class="muted">Used for every version in this project and new CapCut handoffs.</p><p id="project-name-status" role="status"></p><div class="project-dialog-actions"><button type="button" data-close-project-dialog="project-name-dialog" class="secondary">Cancel</button><button type="submit" id="project-name-save">Save name</button></div></form></dialog><dialog id="project-delete-dialog" class="project-dialog" aria-labelledby="project-delete-title"><h2 id="project-delete-title">Delete selected projects?</h2><ul id="project-delete-names"></ul><p id="project-delete-summary"></p><p id="project-delete-kept" class="muted"></p><p id="project-delete-status" role="status"></p><div class="project-dialog-actions"><button type="button" data-close-project-dialog="project-delete-dialog" class="secondary">Cancel</button><button type="button" id="project-delete-recheck" class="secondary">Check again</button><button type="button" id="project-delete-confirm" class="danger" disabled>Delete permanently</button></div></dialog>';
    doc.body.appendChild(modal);
    async function request(url, body, method='POST') {
      const response=await fetcher(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const data=await response.json();
      if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:'Could not complete this action. Please retry.');
      return data;
    }
    function state() { return selection(getJobs(),selected,currentFilter,currentQuery); }
    function sync(filter=currentFilter,query=currentQuery) {
      currentFilter=filter; currentQuery=query;
      const data=state(), keys=new Set(data.all.map(g=>g.key));
      for(const key of selected) if(!keys.has(key)) selected.delete(key);
      el('projects-selected').textContent=`${data.chosen.length} selected`;
      el('projects-clear').disabled=busy||!selected.size;
      el('projects-delete').disabled=busy||!data.chosen.length||data.chosen.some(g=>g.jobs.some(UI.active));
      el('projects-select-all').disabled=busy||!data.matching.some(g=>!g.jobs.some(UI.active));
    }
    el('projects-select-all').addEventListener('click',()=>{state().matching.filter(g=>!g.jobs.some(UI.active)).forEach(g=>selected.add(g.key));render();});
    el('projects-clear').addEventListener('click',()=>{selected.clear();render();});
    el('jobs').addEventListener('change',event=>{
      const input=event.target.closest('[data-project-select]');
      if(!input||busy)return;
      if(input.checked)selected.add(input.dataset.projectSelect);else selected.delete(input.dataset.projectSelect);
      render();
    });
    el('jobs').addEventListener('click',event=>{
      const button=event.target.closest('.btn-project-name');
      if(!button||busy)return;
      renameJob=getJobs().find(j=>j.job_id===button.dataset.jobId);
      if(!renameJob)return;
      el('project-name-input').value=UI.title(renameJob);el('project-name-status').textContent='';
      el('project-name-dialog').showModal();el('project-name-input').focus();el('project-name-input').select();
    });
    el('project-name-form').addEventListener('submit',async event=>{
      event.preventDefault();if(busy)return;busy=true;el('project-name-save').disabled=true;
      try {
        await request(`/api/projects/${encodeURIComponent(renameJob.job_id)}`,{name:el('project-name-input').value.trim()},'PATCH');
        el('project-name-dialog').close();await refresh();
      }catch(error){el('project-name-status').textContent=error.message;}
      finally{busy=false;el('project-name-save').disabled=false;render();}
    });
    async function preview() {
      if(busy)return;busy=true;currentPlan=null;el('project-delete-confirm').disabled=true;
      el('project-delete-recheck').disabled=true;el('project-delete-status').textContent='Checking project resources…';
      try {
        const chosen=state().chosen;
        if(!chosen.length)throw new Error('Select a project first.');
        const ids=chosen.map(g=>g.titleJob.job_id);
        const plan=await request('/api/projects/delete-preview',{job_ids:ids});
        currentPlan={...plan,requested:ids};
        el('project-delete-names').innerHTML=plan.project_names.map(n=>`<li>${UI.escape(n)}</li>`).join('');
        el('project-delete-summary').textContent=summary(plan);
        el('project-delete-kept').textContent=plan.kept_sources.length?`${plan.kept_sources.length} shared or external source videos will be kept because they are used elsewhere or are outside the app's upload folder.`:'';
        el('project-delete-status').textContent=plan.issues.join(' ');
        el('project-delete-confirm').disabled=plan.issues.length>0;
      }catch(error){el('project-delete-status').textContent=error.message;}
      finally{busy=false;el('project-delete-recheck').disabled=false;render();}
    }
    el('projects-delete').addEventListener('click',()=>{
      el('project-delete-names').innerHTML='';el('project-delete-summary').textContent='';el('project-delete-kept').textContent='';
      el('project-delete-dialog').showModal();preview();
    });
    el('project-delete-recheck').addEventListener('click',preview);
    el('project-delete-confirm').addEventListener('click',async()=>{
      if(busy||!currentPlan||currentPlan.issues.length)return;busy=true;el('project-delete-confirm').disabled=true;el('project-delete-recheck').disabled=true;
      el('project-delete-status').textContent='Deleting projects and their resources…';
      try {
        const data=await request('/api/projects/delete',{job_ids:currentPlan.requested,token:currentPlan.token});
        selected.clear();el('project-delete-dialog').close();
        el('project-cleanup').classList.toggle('hidden',!data.cleanup_pending);
        await refresh();root.dispatchEvent(new root.Event('snooker:work-changed'));
      }catch(error){currentPlan=null;el('project-delete-status').textContent=error.message+' Click Check again to review the current resources.';}
      finally{busy=false;el('project-delete-recheck').disabled=false;sync();}
    });
    doc.querySelectorAll('[data-close-project-dialog]').forEach(button=>button.addEventListener('click',()=>{if(!busy)el(button.dataset.closeProjectDialog).close();}));
    [el('project-name-dialog'),el('project-delete-dialog')].forEach(dialog=>dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();}));
    el('projects-cleanup').addEventListener('click',async()=>{
      if(busy)return;busy=true;
      try{const data=await request('/api/projects/cleanup',{});el('project-cleanup').classList.toggle('hidden',!data.cleanup_pending);await refresh();}
      catch(error){const status=el('library-status');status.textContent=error.message;status.classList.remove('hidden');}
      finally{busy=false;render();}
    });
    fetcher('/api/projects/cleanup').then(r=>r.ok?r.json():null).then(data=>{if(data)el('project-cleanup').classList.toggle('hidden',!data.cleanup_pending);}).catch(()=>{});
    return {selected,sync,isBusy:()=>busy};
  }
  const api={create,selection,summary};root.snookerLibraryProjects=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
