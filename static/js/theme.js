// SkillHire Theme Toggle - Fixed Version

(function () {
  function isDashboardPage() {
    return document.body.classList.contains('app-body') || 
           !!document.querySelector('.app-shell') ||
           !!document.querySelector('.sidebar-nav');
  }

  function applyTheme(theme) {
    const dashboard = isDashboardPage();
    if (dashboard) {
      if (theme === 'light') {
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
      } else {
        document.body.classList.remove('light-mode');
        document.body.classList.remove('dark-mode');
      }
    } else {
      if (theme === 'dark') {
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
      } else {
        document.body.classList.remove('dark-mode');
        document.body.classList.remove('light-mode');
      }
    }
    updateToggleLabel(theme, dashboard);
  }

  function updateToggleLabel(theme, dashboard) {
    const btn = document.getElementById('themeToggleBtn');
    if (!btn) return;
    if (dashboard) {
      btn.innerHTML = theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
    } else {
      btn.innerHTML = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    }
  }

  window.toggleTheme = function () {
    const dashboard = isDashboardPage();
    const stored = localStorage.getItem('skillhire_theme') || 'default';
    let next;
    if (dashboard) {
      next = stored === 'light' ? 'dark' : 'light';
    } else {
      next = stored === 'dark' ? 'light' : 'dark';
    }
    localStorage.setItem('skillhire_theme', next);
    applyTheme(next);
  };

  function init() {
    const stored = localStorage.getItem('skillhire_theme') || 'default';
    applyTheme(stored);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
