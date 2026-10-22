/* The same local editor, adapted only for Android storage and desktop actions. */
document.addEventListener('DOMContentLoaded', () => {
  document.querySelector('#dl-workflow-option')?.remove();
  document.querySelectorAll('#workflow option[value="dl_algo"]').forEach((option) => option.remove());
  const note = document.createElement('p');
  note.className = 'mobile-note';
  note.textContent = 'On this phone · Video processing stays on your device';
  document.querySelector('.top')?.after(note);
  const heading = document.querySelector('.page-heading .muted');
  if (heading) heading.textContent = 'Choose a match from your phone. Detect, edit and export your highlights here.';
  function adapt() {
    for (const node of document.querySelectorAll('button, a')) {
      const text = node.textContent.trim();
      if (/CapCut|Open.*folder|Show in.*folder/i.test(text)) node.dataset.mobileDesktop = '';
    }
    const file = document.getElementById('file-name');
    if (file?.textContent === 'Choose videos or drop them here') file.textContent = 'Choose videos from your phone';
    const button = document.getElementById('file-action');
    if (button && button.textContent !== 'Choose videos') button.textContent = 'Choose videos';
  }
  adapt();
  new MutationObserver(adapt).observe(document.body, {childList: true, subtree: true});
});
