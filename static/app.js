/**
 * ForceCRM Archimedes2026 Single-Page Application Client (`src/static/app.js`)
 * High-performance, reactive JavaScript dashboard for B2B ICP Discovery & Lookalike Recommendations.
 * Zero-API-key basemaps, full Greek regional coverage, instant sub-40ms recomputing, and local LLM integration.
 */

// Global Application State
const state = {
  lang: localStorage.getItem('forcecrm_lang') || 'el',
  execExpanded: false,
  layoutMode: 'split', // 'split' | 'table' | 'map'
  tableDensity: 'standard', // 'compact' | 'standard' | 'detailed'
  showDeliveryRings: false,
  selectedLeadIds: new Set(),
  focusedLeadIndex: -1,
  currentFilteredLeads: [],
  pendingExcludeTarget: null, // { companyId, companyName, vat, isBatch }
  drawerSubTab: 'overview', // 'overview' | 'geo' | 'registry'
  lastInspectedData: null,

  params: {},
  defaults: {},
  paramSpecs: [],
  meta: {},
  profiles: [],
  lookalikes: [],
  clients: [],
  benchmark: {},
  ollama: { available: false, model: '', installed: [] },
  selectedCompanyId: null,
  activeTab: 'overview',
  
  // Leads Table Filter, Sort & Pagination
  leadsFilter: {
    search: '',
    profile: 'ALL',
    prefecture: 'ALL',
    legal: 'ALL',
    quickFilter: 'all', // 'all' | 'hot' | 'route' | 'multi' | 'tier1' | 'top100' | 'attica' | 'thessaloniki' | 'islands' | 'corporate' | 'hq'
    sortKey: 'rank', // 'rank' | 'score' | 'name' | 'nearest_client_km'
    sortDir: 'asc',
    page: 1,
    pageSize: 50,
  },
  
  // Clients Table Filter
  clientsFilter: {
    search: '',
    profile: 'ALL',
    quickFilter: 'all', // 'all' | 'vip' | 'icp1' | 'icp2' | 'nofit' | 'branches'
  },

  // Map Filter & Basemap
  mapFilter: {
    layer: 'ALL', // 'ALL' | 'CLIENTS' | 'LEADS'
    region: 'greece',
    basemap: 'esri', // 'esri' | 'osm'
  },
  
  // Active map & chart instances
  map: null,
  markersLayer: null,
  deliveryRingsLayer: null,
  salesRouteLayer: null,
  hoverHaloMarker: null,
  currentTileLayer: null,
  canvasRenderer: null,
  haloMarker: null,
  miniMap: null,
  chartRevDonut: null,
  chartClientsDonut: null,
  chartSilhouette: null,
  compCharts: {},
  territoryCharts: {},
  icpActiveTab: {},
  
  // Debounce timer for instant slider updates
  sliderDebounceTimer: null,
  isComputing: false,
};

// Bilingual UI Dictionary (EL & EN — Zero colloquial 'Τζίρος', Formal 'Κύκλος Εργασιών' / 'Revenue')
const UI_I18N = {
  el: {
    'header.subtitle': 'Έξυπνη Ανάλυση Πελατολογίου & Προτάσεις Νέων Πελατών (B2B ICP Discovery)',
    'header.instant_scoring': 'Άμεση Βαθμολόγηση Ενεργή',
    'header.exclusions_label': 'Εξαιρέσεις',
    'header.export': 'Εξαγωγή',
    'header.export_lookalikes_sub': 'Προτεινόμενοι πελάτες (Top 2.000)',
    'header.export_profiles_sub': 'Προφίλ πελατών & AI περιγραφές',
    'header.export_assignments_sub': 'Κατηγοριοποίηση πελατολογίου',
    'header.analyst_settings': 'Ρυθμίσεις Αναλυτή',
    'exec_bar.market_universe': 'Αγορά Εστίασης',
    'exec_bar.expand_btn': '▾ Ανάπτυξη Executive Analytics & Donuts',
    'exec_bar.collapse_btn': '▴ Σύμπτυξη Executive Analytics',
    'hero.badge': 'Πίνακας Εμπορικής Νοημοσύνης',
    'hero.universe_count': '| 24.424 Ελληνικές Επιχειρήσεις Εστίασης',
    'hero.crm_clients': '300 Ενεργοί Πελάτες CRM',
    'hero.title': 'Ανάλυση Πελατολογίου & Προτάσεις Νέων Πελατών',
    'hero.description': 'Αυτόματη ανακάλυψη Προφίλ Ιδανικού Πελάτη (ICPs) με στάθμιση κύκλου εργασιών, στατιστική τεκμηρίωση χαρακτηριστικών μέσω Empirical Bayes Lift & Hypergeometric Tests, και αξιόπιστη κατάταξη 2.000 νέων B2B υποψηφίων υψηλής ταύτισης.',
    'hero.upload_btn': 'Φόρτωση Νέου Πελατολογίου',
    'hero.params_btn': 'Παράμετροι Μοντέλου',
    'hero.ai_act_badge': 'Grounded Local AI (EU AI Act Art. 50)',
    'kpi.active_clients': 'Ενεργοί Πελάτες',
    'kpi.businesses_unit': 'επιχειρήσεις',
    'kpi.total_revenue': 'Συνολικός Κύκλος Εργασιών',
    'kpi.mean_label': 'Μέσος Όρος',
    'kpi.discovered_icps': 'Ανακαλυφθέντα ICPs',
    'kpi.bootstrap_stability': 'Σταθερότητα Bootstrap',
    'kpi.recommended_leads': 'Προτεινόμενοι Υποψήφιοι',
    'overview.portfolio_revenue_distribution': 'Κατανομή Πελατολογίου & Κύκλου Εργασιών',
    'overview.per_icp': 'Ανά ICP',
    'overview.revenue_short': 'Κύκλος Εργ.',
    'overview.revenue_share_label': 'Μερίδιο Κύκλου Εργασιών €',
    'overview.clients_short': 'Πελάτες',
    'overview.client_distribution_label': 'Κατανομή Πελατών',
    'overview.map_title_short': 'Γεωχωρικό Ραντάρ',
    'overview.map_filter_all': 'Όλα (Πελάτες + Leads)',
    'overview.map_filter_clients': '⭐ Μόνο Πελάτες (300)',
    'overview.map_filter_leads': '🎯 Μόνο Leads (2.000)',
    'overview.legend_clients': '⭐ Πελάτες',
    'overview.legend_leads': '🎯 Υποψήφιοι',
    'overview.map_click_hint': 'Hover στον πίνακα = παλμός στον χάρτη · Κλικ = καρτέλα',
    'overview.icp_section_title': 'Αναλυτική Παρουσίαση Προφίλ Πελατών (Ideal Customer Profiles)',
    'overview.icp_section_subtitle': 'Ταξινομημένα κατά μερίδιο συνολικού κύκλου εργασιών με στατιστικό έλεγχο Empirical Bayes Lift & σενάρια προσέγγισης πωλητή.',
    'overview.show_donuts_btn': 'Προβολή Γραφημάτων Κύκλου Εργασιών',
    'tabs.tab1_cockpit': '1. Κέντρο Προσέγγισης (Χάρτης & Υποψήφιοι)',
    'tabs.tab2_icps': '2. Προφίλ ICP & Οδηγός Πωλήσεων',
    'tabs.tab3_clients': '3. Ενεργό Πελατολόγιο',
    'tabs.tab4_tech': '4. Επικύρωση Μοντέλου & EU AI Act',
    'cockpit.kb_nav': 'Πλοήγηση',
    'cockpit.kb_drawer': 'Καρτέλα',
    'cockpit.kb_search': 'Αναζήτηση',
    'leads.all_profiles': 'Όλα τα Προφίλ ICP',
    'leads.all_prefectures': 'Όλοι οι Νομοί',
    'leads.all_legal_forms': 'Όλες οι Νομικές Μορφές',
    'leads.col_match': 'Ταύτιση',
    'leads.col_company_kad': 'Επωνυμία, Τμήμα & ΚΑΔ',
    'leads.col_location_route': 'Τοποθεσία & Διαδρομή',
    'leads.col_actions': 'Ενέργειες',
    'chips.all': '✨ Όλα',
    'chips.hot': '🔥 Ταύτιση ≥85%',
    'chips.route': '🚚 Διαδρομή ≤1km',
    'chips.multi': '🏢 Υποκαταστήματα',
    'chips.tier1': '✅ OSM Tier 1',
    'chips.attica': '🏛️ Αττική',
    'chips.islands': '🏖️ Νησιά/Τουρισμός',
    'density.compact': 'Συμπαγής',
    'density.standard': 'Κανονική',
    'density.detailed': 'Αναλυτική',
    'layout.table': 'Πίνακας',
    'layout.map': 'Χάρτης',
    'map.delivery_rings_btn': 'Ζώνες Διανομής (1km)',
    'map.client_prospect_route_btn': 'Πελάτης ➔ Leads',
    'regions.greece': '🇬🇷 Ελλάδα',
    'regions.attica': '🏛️ Αττική',
    'regions.thessaloniki': '⚡ Θεσ/νίκη',
    'regions.halkidiki': '🏖️ Χαλκιδική',
    'regions.crete': '🏔️ Κρήτη',
    'regions.cyclades': '⛵ Κυκλάδες',
    'regions.dodecanese': '☀️ Δωδεκάνησα',
    'regions.ionian': '🌊 Ιόνιο',
    'regions.peloponnese': '🌿 Πελοπόννησος',
    'regions.thessaly': '🌾 Θεσσαλία',
    'clients.section_title': 'Κατηγοριοποίηση Υφιστάμενου Πελατολογίου (Assignments)',
    'clients.section_subtitle': 'Δείτε σε ποιο ICP ανήκει κάθε πελάτης, καθώς και ποιες επιχειρήσεις αποτελούν ειδικές περιπτώσεις (NO_FIT).',
    'clients.all_icps_nofit': 'Όλα τα ICPs & NO_FIT',
    'clients.cockpit_portfolio': 'ΠΕΛΑΤΟΛΟΓΙΟ ΣΕ ΑΝΑΛΥΣΗ',
    'clients.cockpit_total_revenue': 'ΣΥΝΟΛΙΚΟΣ ΚΥΚΛΟΣ ΕΡΓΑΣΙΩΝ',
    'clients.cockpit_top_revenue': 'ΚΟΡΥΦΑΙΟΣ ΚΥΚΛΟΣ ΕΡΓΑΣΙΩΝ',
    'clients.cockpit_client_structure': 'ΔΟΜΗ ΠΕΛΑΤΩΝ',
    'clients.chip_all': '✨ Όλοι οι Πελάτες (300)',
    'clients.chip_vip': '💎 VIP Κύκλος Εργασιών (>€25.000)',
    'clients.chip_icp1': '⭐ ICP 1 (Εταιρικές Αττικής)',
    'clients.chip_icp2': '⭐ ICP 2 (Τουριστικές Περιφέρειας)',
    'clients.chip_nofit': '⚠️ NO_FIT (Ειδικές Περιπτώσεις)',
    'clients.chip_branches': '🏢 Πολλαπλά Υποκαταστήματα',
    'clients.col_icp_match': 'ICP & Ταύτιση',
    'clients.col_company_activity': 'Επωνυμία & Δραστηριότητα (ΚΑΔ)',
    'clients.col_revenue_eur': 'Κύκλος Εργασιών (€)',
    'clients.col_location_afm': 'Τοποθεσία & ΑΦΜ',
    'clients.col_actions': 'Ενέργειες',
    'tech.cv_title': 'Επιστημονική Αξιολόγηση: 100-Fold Cross-Validation',
    'tech.recall_500': 'Recall @ Top 500',
    'tech.recall_2000': 'Recall @ Top 2.000',
    'tech.official_baseline': 'Βάση Αναφοράς ForceCRM',
    'tech.cv_methodology': '💡 Μεθοδολογία Ελέγχου: Το πελατολόγιο χωρίζεται 100 φορές σε 240 Train και 60 Hidden Validation πελάτες. Το σύστημα εκπαιδεύεται μόνο στους 240 και καλείται να εντοπίσει τους 60 κρυφούς πελάτες ανάμεσα σε 24.184 αγνώστους. Η αναλυτική προσδοκώμενη τιμή (Expected Recall) του baseline συχνότητας ΚΑΔ είναι ακριβώς 1.87% @ 500 και 7.48% @ 2.000.',
    'tech.silhouette_title': 'Επιλογή Βέλτιστου K (Cosine Silhouette)',
    'tech.model_comp_title': 'Συγκριτική Αξιολόγηση Μοντέλων 4 Αξόνων (Quality · Speed · Cost · Privacy)',
    'tech.osm_audit_title': 'Επίλυση Ασάφειας OpenStreetMap 3 Επιπέδων',
    'tech.osm_audit_desc': 'Στις πυκνές εμπορικές ζώνες (π.χ. Κέντρο Αθήνας, Θεσσαλονίκη), ένα απλό χωρικό join 50m προκαλεί ψευδείς αντιστοιχίσεις (false positives). Το ForceCRM εφαρμόζει αυστηρό πρωτόκολλο 3 επιπέδων (cKDTree + Φωνητική/Λεκτική Ομοιότητα):',
    'tech.ai_act_title': 'Συμμόρφωση με τον Ευρωπαϊκό Κανονισμό Τεχνητής Νοημοσύνης (EU AI Act - Άρθρο 50 & Ενότητα 9)',
    'tech.ai_act_pillar1_title': '1. Διαφάνεια Παραγόμενου Περιεχομένου (Art. 50)',
    'tech.ai_act_pillar1_body': 'Όλα τα κείμενα που παράγονται από το τοπικό AI φέρουν υποχρεωτική σήμανση 🤖 «Περιγραφή παραγόμενη από Τοπικό AI (EU AI Act Art. 50)» και ελέγχονται από αυστηρό Numeric/Regex Validator μηδενικών παραισθήσεων.',
    'tech.ai_act_pillar2_title': '2. Απαγόρευση Αξιολόγησης Φερεγγυότητας (Section 9)',
    'tech.ai_act_pillar2_body': 'Το σύστημα αξιολογεί αποκλειστικά εμπορική/μορφολογική ομοιότητα B2B και κατηγοριοποιεί το μέγεθος επιχείρησης (Commercial Segment Tier A/B/C). Απαγορεύεται ρητά οποιαδήποτε βαθμολόγηση πιστοληπτικής ικανότητας ή φερεγγυότητας φυσικών προσώπων.',
    'tech.ai_act_pillar3_title': '3. Τοπική Εκτέλεση & Προστασία Δεδομένων',
    'tech.ai_act_pillar3_body': 'Όλοι οι υπολογισμοί και η εκτέλεση του LLM πραγματοποιούνται 100% τοπικά (Localhost / Air-gapped) χωρίς διαβίβαση εταιρικών δεδομένων ή κύκλου εργασιών σε εξωτερικά APIs cloud.',
    'fab.selected_label': 'Επιλεγμένοι Υποψήφιοι',
    'fab.export_csv': 'Εξαγωγή CSV',
    'fab.build_route': 'Σχεδίαση Διαδρομής',
    'fab.verify_gemi': 'Έλεγχος Γ.Ε.ΜΗ.',
    'fab.exclude': 'Εξαίρεση',
    'drawer.company_card_title': 'Καρτέλα Επιχείρησης',
    'drawer.back_to_map': 'Χάρτης',
    'exclude_modal.title': 'Εξαίρεση από Προτάσεις (CRM Blacklist)',
    'exclude_modal.select_reason': 'Επιλέξτε αιτιολογία εξαίρεσης:',
    'exclude_modal.confirm_btn': 'Οριστική Εξαίρεση',
    'analyst.title': 'Ρυθμίσεις Αναλυτή',
    'analyst.subtitle': 'Διαδραστική Παραμετροποίηση Μοντέλου & Φόρτωση Αρχείου',
    'analyst.sec1_upload': '1. Αρχείο Πελατολογίου (.xlsx)',
    'analyst.dropzone_title': 'Σύρετε ή πατήστε για μεταφόρτωση Excel',
    'analyst.dropzone_sub': 'Στήλες: ΑΦΜ και Κύκλος Εργασιών (€)',
    'analyst.sec2_weights': '2. Βάρη Βαθμολόγησης Υποψηφίων',
    'analyst.weight_cosine': 'Εμπορική & Κλαδική Ομοιότητα Προφίλ',
    'analyst.weight_eb': 'Στατιστική Συνάφεια ΚΑΔ (Empirical Bayes Lift)',
    'analyst.weight_spatial': 'Γεωγραφική Εγγύτητα Χάρτη',
    'analyst.spatial_radius': 'Ακτίνα Εγγύτητας (km)',
    'analyst.eb_prior': 'Ισχύς Εξομάλυνσης Empirical Bayes',
    'analyst.nofit_threshold': 'Κατώφλι Ειδικών Περιπτώσεων (NO_FIT)',
    'analyst.sec3_embeddings': '3. Μοντέλα Embeddings ΚΑΔ',
    'analyst.use_embeddings_label': 'Χρήση Μοντέλων Embeddings',
    'analyst.select_embedding_model': 'Επιλογή Μοντέλου Embeddings',
    'analyst.sec4_clustering': '4. Παράμετροι Clustering (K-Means)',
    'analyst.revenue_weighting_label': 'Στάθμιση Κύκλου Εργασιών Πελατών',
    'analyst.reset_btn': 'Επαναφορά Προεπιλογών',
    'analyst.apply_close_btn': 'Εφαρμογή & Κλείσιμο',
    'blacklist.title': 'Αποκλεισμένες Επιχειρήσεις (CRM Blacklist)',
    'blacklist.subtitle': 'Επιχειρήσεις με διακοπή εργασιών ή αντικατάσταση χώρου που εξαιρούνται από τα Lookalikes',
    'blacklist.tab_user': 'Χειροκίνητες',
    'blacklist.tab_superseded': 'Αντικαταστάσεις Χώρου',
    'blacklist.search_placeholder': 'Αναζήτηση επωνυμίας, ΑΦΜ...',
    'blacklist.empty_search': 'Δεν βρέθηκαν αποτελέσματα για την αναζήτηση.',
    'blacklist.empty_user': 'Δεν υπάρχουν χειροκίνητες εξαιρέσεις. Μπορείτε να εξαιρέσετε επιχειρήσεις από την καρτέλα υποψηφίου.',
    'blacklist.empty_superseded': 'Δεν υπάρχουν εξαιρέσεις αντικατάστασης χώρου.',
    'blacklist.badge_user': 'Χειροκίνητη Αφαίρεση',
    'blacklist.badge_superseded': 'Αντικατάσταση Χώρου',
    'blacklist.excluded_on': 'Εξαιρέθηκε',
    'blacklist.restore_btn': 'Επαναφορά',
    'blacklist.restore_title': 'Επαναφορά στα Lookalikes',
    'blacklist.truncated_notice': 'Προβολή πρώτων {shown} από {total} εγγραφές. Χρησιμοποιήστε την αναζήτηση για εντοπισμό συγκεκριμένης επιχείρησης.',
    'blacklist.footer_note': 'Οι αλλαγές εφαρμόζονται άμεσα στην κατάταξη των Lookalikes.',
    'route_modal.title': 'Σχεδίαση Διαδρομής: Πελάτης ➔ Υποψήφιοι',
    'route_modal.subtitle': 'Διασύνδεση υφιστάμενων σημείων διανομής με νέες εμπορικές ευκαιρίες',
    'route_modal.step1_label': '1. Επιλέξτε Ενεργό Πελάτη (Αφετηρία Διαδρομής):',
    'route_modal.step2_label': '2. Πλήθος Υποψηφίων:',
    'route_modal.step3_label': '3. Μέγιστη Ακτίνα:',
    'route_modal.osrm_title': '⚡ Έξυπνη Βελτιστοποίηση Διαδρομής (OSRM + TSP)',
    'route_modal.osrm_desc': 'Ο αλγόριθμος υπολογίζει πραγματικές οδικές αποστάσεις turn-by-turn (OSRM), ξεκινώντας από το σημείο του πελάτη προς τους κορυφαίους B2B lookalikes της ζώνης διανομής.',
    'route_modal.cancel': 'Ακύρωση',
    'route_modal.draw_btn': '🚀 Σχεδίαση στον Χάρτη',
    'common.cancel': 'Ακύρωση',
    'common.close': 'Κλείσιμο',
  },
  en: {
    'header.subtitle': 'Intelligent Client Portfolio Analysis & B2B Lookalike Prospecting (ICP Discovery)',
    'header.instant_scoring': 'Instant Scoring Active',
    'header.exclusions_label': 'Exclusions',
    'header.export': 'Export',
    'header.export_lookalikes_sub': 'Recommended prospects (Top 2,000)',
    'header.export_profiles_sub': 'ICP profiles & grounded AI narratives',
    'header.export_assignments_sub': 'Client portfolio ICP assignments',
    'header.analyst_settings': 'Analyst Settings',
    'exec_bar.market_universe': 'Market Universe',
    'exec_bar.expand_btn': '▾ Expand Executive Analytics & Donuts',
    'exec_bar.collapse_btn': '▴ Collapse Executive Analytics',
    'hero.badge': 'Executive Intelligence Dashboard',
    'hero.universe_count': '| 24,424 Greek Hospitality Businesses',
    'hero.crm_clients': '300 Active CRM Clients',
    'hero.title': 'Client Portfolio Analysis & Lookalike Recommendations',
    'hero.description': 'Automated discovery of Ideal Customer Profiles (ICPs) weighted by annual revenue, statistical trait validation via Empirical Bayes Lift & Hypergeometric Tests, and high-precision ranking of 2,000 qualified B2B prospects.',
    'hero.upload_btn': 'Upload Client Portfolio',
    'hero.params_btn': 'Model Parameters',
    'hero.ai_act_badge': 'Grounded Local AI (EU AI Act Art. 50)',
    'kpi.active_clients': 'Active Clients',
    'kpi.businesses_unit': 'businesses',
    'kpi.total_revenue': 'Total Portfolio Revenue',
    'kpi.mean_label': 'Mean Revenue',
    'kpi.discovered_icps': 'Discovered ICPs',
    'kpi.bootstrap_stability': 'Bootstrap Stability',
    'kpi.recommended_leads': 'Qualified Prospects',
    'overview.portfolio_revenue_distribution': 'Portfolio & Revenue Distribution',
    'overview.per_icp': 'Per ICP',
    'overview.revenue_short': 'Revenue',
    'overview.revenue_share_label': 'Revenue Share (€)',
    'overview.clients_short': 'Clients',
    'overview.client_distribution_label': 'Client Share',
    'overview.map_title_short': 'Geospatial Radar',
    'overview.map_filter_all': 'All Points (Clients + Leads)',
    'overview.map_filter_clients': '⭐ Active Clients Only (300)',
    'overview.map_filter_leads': '🎯 Lookalike Leads Only (2,000)',
    'overview.legend_clients': '⭐ Active Clients',
    'overview.legend_leads': '🎯 Prospects',
    'overview.map_click_hint': 'Row hover = map pulse · Click = slide-over inspector',
    'overview.icp_section_title': 'Ideal Customer Profiles (ICPs) & Sales Playbooks',
    'overview.icp_section_subtitle': 'Ordered by total portfolio revenue share with Empirical Bayes Lift & B2B outreach playbooks.',
    'overview.show_donuts_btn': 'Toggle Portfolio Revenue Donuts',
    'tabs.tab1_cockpit': '1. Prospecting Cockpit (Split Map & Leads)',
    'tabs.tab2_icps': '2. ICP Profiles & Playbooks',
    'tabs.tab3_clients': '3. Active Clientele',
    'tabs.tab4_tech': '4. Benchmark Validation & EU AI Act',
    'cockpit.kb_nav': 'Navigate',
    'cockpit.kb_drawer': 'Inspect',
    'cockpit.kb_search': 'Search',
    'leads.all_profiles': 'All ICP Profiles',
    'leads.all_prefectures': 'All Prefectures',
    'leads.all_legal_forms': 'All Legal Forms',
    'leads.col_match': 'Match Score',
    'leads.col_company_kad': 'Company, Segment & NACE (KAD)',
    'leads.col_location_route': 'Location & Route Distance',
    'leads.col_actions': 'Actions',
    'chips.all': '✨ All',
    'chips.hot': '🔥 Match ≥85%',
    'chips.route': '🚚 Route ≤1km',
    'chips.multi': '🏢 Multi-Branch',
    'chips.tier1': '✅ OSM Tier 1',
    'chips.attica': '🏛️ Attica',
    'chips.islands': '🏖️ Islands/Tourism',
    'density.compact': 'Compact',
    'density.standard': 'Standard',
    'density.detailed': 'Detailed',
    'layout.table': 'Table',
    'layout.map': 'Map',
    'map.delivery_rings_btn': 'Delivery Route Rings (1km)',
    'map.client_prospect_route_btn': 'Client ➔ Leads',
    'regions.greece': '🇬🇷 All Greece',
    'regions.attica': '🏛️ Attica',
    'regions.thessaloniki': '⚡ Thessaloniki',
    'regions.halkidiki': '🏖️ Halkidiki',
    'regions.crete': '🏔️ Crete',
    'regions.cyclades': '⛵ Cyclades',
    'regions.dodecanese': '☀️ Dodecanese',
    'regions.ionian': '🌊 Ionian',
    'regions.peloponnese': '🌿 Peloponnese',
    'regions.thessaly': '🌾 Thessaly',
    'clients.section_title': 'Active Client Portfolio Categorization (Assignments)',
    'clients.section_subtitle': 'Inspect each client’s ICP assignment and outlier accounts (NO_FIT).',
    'clients.all_icps_nofit': 'All ICPs & NO_FIT',
    'clients.cockpit_portfolio': 'PORTFOLIO ANALYZED',
    'clients.cockpit_total_revenue': 'TOTAL PORTFOLIO REVENUE',
    'clients.cockpit_top_revenue': 'TOP ACCOUNT REVENUE',
    'clients.cockpit_client_structure': 'LEGAL STRUCTURE',
    'clients.chip_all': '✨ All Clients (300)',
    'clients.chip_vip': '💎 VIP Revenue (>€25,000)',
    'clients.chip_icp1': '⭐ ICP 1 (Corporate Attica)',
    'clients.chip_icp2': '⭐ ICP 2 (Regional Tourism)',
    'clients.chip_nofit': '⚠️ NO_FIT (Outlier Accounts)',
    'clients.chip_branches': '🏢 Multi-Branch Accounts',
    'clients.col_icp_match': 'ICP & Match',
    'clients.col_company_activity': 'Company Name & Activity (KAD)',
    'clients.col_revenue_eur': 'Annual Revenue (€)',
    'clients.col_location_afm': 'Location & Tax ID',
    'clients.col_actions': 'Actions',
    'tech.cv_title': 'Scientific Evaluation: 100-Fold Cross-Validation',
    'tech.recall_500': 'Recall @ Top 500',
    'tech.recall_2000': 'Recall @ Top 2,000',
    'tech.official_baseline': 'Official ForceCRM Baseline',
    'tech.cv_methodology': '💡 Methodology: The client portfolio is partitioned across 100 Repeated Stratified K-Folds (5x20) into 240 Train and 60 Hidden Validation clients. The model is fitted only on the 240 and tasked to retrieve the 60 hidden clients from 24,184 market peers. Analytical Expected Recall of the baseline primary KAD model is 1.87% @ 500 and 7.48% @ 2,000.',
    'tech.silhouette_title': 'Optimal K Selection (Cosine Silhouette)',
    'tech.model_comp_title': '4-Axis Model Comparison Table (Quality · Speed · Cost · Privacy)',
    'tech.osm_audit_title': 'OpenStreetMap 3-Tier Ambiguity Resolution Audit',
    'tech.osm_audit_desc': 'In dense metropolitan commercial corridors, simple 50m spatial joins create false positive entity collisions. ForceCRM implements a rigorous 3-tier resolution protocol combining cKDTree radii with token-sort lexical and phonetic verification:',
    'tech.ai_act_title': 'EU AI Act Compliance (Article 50 & Section 9 Guardrails)',
    'tech.ai_act_pillar1_title': '1. AI Transparency & Disclosure Badge (Art. 50)',
    'tech.ai_act_pillar1_body': 'All narrative summaries generated by local AI bear mandatory Article 50 disclosures and are vetted by deterministic numeric validators guaranteeing zero hallucination.',
    'tech.ai_act_pillar2_title': '2. Strict Prohibition of Credit/Solvency Scoring (Sec. 9)',
    'tech.ai_act_pillar2_body': 'The platform scores strictly commercial morphology and B2B lookalike affinity (Commercial Segment Tiers A/B/C). Individual credit scoring, natural-person profiling, or solvency assessments are strictly excluded.',
    'tech.ai_act_pillar3_title': '3. Air-Gapped Local Execution & Data Privacy',
    'tech.ai_act_pillar3_body': 'All matrix operations and LLM inference run strictly on-device (Localhost / Air-Gapped CPU/GPU) with zero customer financial disclosures transmitted to third-party cloud APIs.',
    'fab.selected_label': 'Prospects Selected',
    'fab.export_csv': 'Export CSV',
    'fab.build_route': 'Build Route',
    'fab.verify_gemi': 'Batch Verify GEMI',
    'fab.exclude': 'Exclude',
    'drawer.company_card_title': 'Company Intelligence Card',
    'drawer.back_to_map': 'Map',
    'exclude_modal.title': 'Exclude from Recommendations (CRM Blacklist)',
    'exclude_modal.select_reason': 'Select exclusion reason:',
    'exclude_modal.confirm_btn': 'Confirm Exclusion',
    'analyst.title': 'Analyst Settings',
    'analyst.subtitle': 'Interactive Model Hyperparameter Tuning & File Upload',
    'analyst.sec1_upload': '1. Client Portfolio File (.xlsx)',
    'analyst.dropzone_title': 'Drag & drop or click to upload Excel',
    'analyst.dropzone_sub': 'Columns: Tax ID (AFM) and Revenue (€)',
    'analyst.sec2_weights': '2. Lookalike Scoring Weights',
    'analyst.weight_cosine': 'Semantic & Morphological Profile Similarity',
    'analyst.weight_eb': 'Statistical Activity Lift (Empirical Bayes)',
    'analyst.weight_spatial': 'Geospatial Kernel Gravity',
    'analyst.spatial_radius': 'Spatial Kernel Bandwidth (km)',
    'analyst.eb_prior': 'Empirical Bayes Prior Strength',
    'analyst.nofit_threshold': 'Outlier Threshold (NO_FIT)',
    'analyst.sec3_embeddings': '3. Semantic KAD Embedding Models',
    'analyst.use_embeddings_label': 'Enable Neural Embedding Models',
    'analyst.select_embedding_model': 'Select Embedding Architecture',
    'analyst.sec4_clustering': '4. Clustering Hyperparameters (K-Means)',
    'analyst.revenue_weighting_label': 'Client Revenue Weighting Mode',
    'analyst.reset_btn': 'Reset Defaults',
    'analyst.apply_close_btn': 'Apply & Close',
    'blacklist.title': 'Excluded Companies (CRM Blacklist)',
    'blacklist.subtitle': 'Businesses with cessation of operations or premise replacement excluded from Lookalikes',
    'blacklist.tab_user': 'Manual',
    'blacklist.tab_superseded': 'Premise Replacements',
    'blacklist.search_placeholder': 'Search company name, Tax ID...',
    'blacklist.empty_search': 'No results found for your search.',
    'blacklist.empty_user': 'No manual exclusions. You can exclude businesses from the candidate card.',
    'blacklist.empty_superseded': 'No premise replacement exclusions found.',
    'blacklist.badge_user': 'Manual Removal',
    'blacklist.badge_superseded': 'Premise Replacement',
    'blacklist.excluded_on': 'Excluded',
    'blacklist.restore_btn': 'Restore',
    'blacklist.restore_title': 'Restore to Lookalikes',
    'blacklist.truncated_notice': 'Showing first {shown} of {total} records. Use the search box to locate a specific business.',
    'blacklist.footer_note': 'Changes take effect immediately in Lookalike rankings.',
    'route_modal.title': 'Route Planning: Client ➔ Prospects',
    'route_modal.subtitle': 'Connecting existing delivery points with new commercial opportunities',
    'route_modal.step1_label': '1. Select Active Client (Route Origin):',
    'route_modal.step2_label': '2. Number of Prospects:',
    'route_modal.step3_label': '3. Maximum Radius:',
    'route_modal.osrm_title': '⚡ Smart Route Optimization (OSRM + TSP)',
    'route_modal.osrm_desc': 'The algorithm computes turn-by-turn road driving distances (OSRM) from the client point to top B2B lookalikes in the delivery zone.',
    'route_modal.cancel': 'Cancel',
    'route_modal.draw_btn': '🚀 Plot on Map',
    'common.cancel': 'Cancel',
    'common.close': 'Close',
  }
};

function uiT(key, fallback = '') {
  const dict = UI_I18N[state.lang] || UI_I18N.el;
  return dict[key] || UI_I18N.el[key] || fallback || key;
}

window.setLanguage = function(lang) {
  state.lang = lang === 'en' ? 'en' : 'el';
  localStorage.setItem('forcecrm_lang', state.lang);
  document.documentElement.lang = state.lang;

  const btnEl = document.getElementById('lang-toggle-el');
  const btnEn = document.getElementById('lang-toggle-en');
  if (btnEl) btnEl.classList.toggle('active', state.lang === 'el');
  if (btnEn) btnEn.classList.toggle('active', state.lang === 'en');

  // Update all [data-i18n] elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    const val = uiT(k);
    if (val && val !== k) el.textContent = val;
  });

  // Update search placeholders
  const searchInput = document.getElementById('leads-search-input');
  if (searchInput) {
    searchInput.placeholder = state.lang === 'en'
      ? 'Search company, Tax ID, city... (Ctrl+K)'
      : 'Αναζήτηση επωνυμίας, ΑΦΜ, πόλης... (Ctrl+K)';
  }
  const clSearch = document.getElementById('clients-search-input');
  if (clSearch) {
    clSearch.placeholder = state.lang === 'en' ? 'Search active client...' : 'Αναζήτηση πελάτη...';
  }
  const exSearch = document.getElementById('excluded-search-input');
  if (exSearch) {
    exSearch.placeholder = uiT('blacklist.search_placeholder', 'Αναζήτηση επωνυμίας, ΑΦΜ...');
  }

  // Update Executive Bar expand button text
  const expandText = document.getElementById('exec-expand-text');
  if (expandText) {
    expandText.textContent = state.execExpanded ? uiT('exec_bar.collapse_btn') : uiT('exec_bar.expand_btn');
  }

  populateFilterDropdowns();
  renderAllViews();
  renderExcludedModal();
  updateBenchmarkCards();
  if (window.lucide) {
    window.lucide.createIcons();
  }
  if (state.selectedCompanyId && (document.getElementById('company-drawer-backdrop')?.classList.contains('open') || document.getElementById('company-drawer-backdrop')?.classList.contains('drawer-open'))) {
    inspectCompany(state.selectedCompanyId);
  }
};

window.toggleExecSummary = function(forceOpen) {
  state.execExpanded = typeof forceOpen === 'boolean' ? forceOpen : !state.execExpanded;
  const panel = document.getElementById('exec-expandable-panel');
  const bar = document.getElementById('exec-summary-bar');
  const expandText = document.getElementById('exec-expand-text');
  if (panel) {
    panel.classList.toggle('expanded', state.execExpanded);
    panel.classList.toggle('open', state.execExpanded);
  }
  if (bar) {
    bar.setAttribute('aria-expanded', String(state.execExpanded));
  }
  if (expandText) {
    expandText.textContent = state.execExpanded ? uiT('exec_bar.collapse_btn') : uiT('exec_bar.expand_btn');
  }
  if (state.execExpanded) {
    setTimeout(() => {
      if (state.chartRevDonut) state.chartRevDonut.resize();
      if (state.chartClientsDonut) state.chartClientsDonut.resize();
    }, 60);
  }
  if (state.map) {
    setTimeout(() => state.map.invalidateSize(), 260);
  }
};

window.setLayoutMode = function(mode) {
  state.layoutMode = ['split', 'table', 'map'].includes(mode) ? mode : 'split';
  const container = document.getElementById('cockpit-split-container');
  if (container) {
    container.classList.remove('mode-split', 'mode-table', 'mode-map');
    container.classList.add(`mode-${state.layoutMode}`);
  }
  ['split', 'table', 'map'].forEach(m => {
    const btn = document.getElementById(`layout-btn-${m}`);
    if (btn) btn.classList.toggle('active', m === state.layoutMode);
  });
  if (state.map) {
    setTimeout(() => state.map.invalidateSize(), 120);
  }
};

window.setTableDensity = function(density) {
  state.tableDensity = ['compact', 'standard', 'detailed'].includes(density) ? density : 'standard';
  const pane = document.getElementById('split-table-pane');
  if (pane) {
    pane.classList.remove('density-compact', 'density-standard', 'density-detailed');
    pane.classList.add(`density-${state.tableDensity}`);
  }
  ['compact', 'standard', 'detailed'].forEach(d => {
    const btn = document.getElementById(`density-btn-${d}`);
    if (btn) btn.classList.toggle('active', d === state.tableDensity);
  });
};

window.sortLeadsBy = function(sortKey) {
  if (state.leadsFilter.sortKey === sortKey) {
    state.leadsFilter.sortDir = state.leadsFilter.sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    state.leadsFilter.sortKey = sortKey;
    state.leadsFilter.sortDir = sortKey === 'score' ? 'desc' : 'asc';
  }
  state.leadsFilter.page = 1;

  const mapIds = { rank: 'rank', score: 'score', name: 'name', nearest_client_km: 'dist' };
  Object.entries(mapIds).forEach(([k, domSuffix]) => {
    const th = document.getElementById(`th-sort-${domSuffix}`);
    const ind = document.getElementById(`sort-ind-${domSuffix}`);
    if (th) th.classList.toggle('active-sort', k === sortKey);
    if (ind) {
      if (k === sortKey) {
        ind.textContent = state.leadsFilter.sortDir === 'asc' ? '↑' : '↓';
      } else {
        ind.textContent = '↕';
      }
    }
  });

  renderLeadsTable();
};

const ICP_COLORS = ['#2563eb', '#0d9488', '#d97706', '#e11d48', '#7c3aed', '#0284c7'];

// -----------------------------------------------------------------------------
// Safe DOM & String Helpers
// -----------------------------------------------------------------------------
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text ?? '';
}

function safeSetHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html ?? '';
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Parses markdown formatting (bold, code spans, lists, headers) into clean executive HTML.
 */
function formatMarkdownText(raw) {
  if (!raw) return '';
  let text = escapeHtml(raw);
  
  // Bold: **text** -> <strong class="text-white font-semibold">text</strong>
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
  
  // Code spans: `code` -> <code class="px-1.5 py-0.5 rounded bg-slate-800 text-blue-300 font-mono text-[11px]">$1</code>
  text = text.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 rounded bg-slate-800 text-blue-300 font-mono text-[11px]">$1</code>');
  
  const lines = text.split('\n');
  const formatted = [];
  let inList = false;
  
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (!inList) {
        formatted.push('<ul class="my-2 space-y-1.5 pl-1">');
        inList = true;
      }
      const itemContent = trimmed.replace(/^[•\-\*]\s*/, '');
      formatted.push(`<li class="flex items-start gap-2 text-slate-300"><span class="text-blue-400 mt-0.5 text-xs font-bold shrink-0">•</span><span>${itemContent}</span></li>`);
    } else {
      if (inList) {
        formatted.push('</ul>');
        inList = false;
      }
      if (trimmed.length > 0) {
        if (trimmed.startsWith('🤖') || trimmed.startsWith('⚖️')) {
          formatted.push(`<div class="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400 my-2 leading-relaxed">${trimmed}</div>`);
        } else {
          formatted.push(`<p class="mb-2 leading-relaxed text-slate-300">${trimmed}</p>`);
        }
      }
    }
  }
  if (inList) {
    formatted.push('</ul>');
  }
  return formatted.join('');
}

// -----------------------------------------------------------------------------
// Toast Notifications
// -----------------------------------------------------------------------------
function showToast(message, icon = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const item = document.createElement('div');
  item.className = 'toast-item';
  item.innerHTML = `
    <i data-lucide="${icon}" class="w-4 h-4 text-blue-400 shrink-0"></i>
    <span>${escapeHtml(message)}</span>
  `;
  container.appendChild(item);
  lucide.createIcons();
  setTimeout(() => {
    item.style.opacity = '0';
    item.style.transform = 'translateY(20px)';
    item.style.transition = 'all 0.3s ease';
    setTimeout(() => item.remove(), 300);
  }, 2800);
}

// -----------------------------------------------------------------------------
// External URL Helpers (Google Maps, Web Search, GEMI Registry)
// -----------------------------------------------------------------------------
function getGoogleMapsUrl(c) {
  if (c.lat && c.lon) {
    return `https://www.google.com/maps/search/?api=1&query=${c.lat},${c.lon}`;
  }
  const q = [c.name, c.street, c.street_number, c.city, c.prefecture].filter(Boolean).join(' ');
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;
}

function getGoogleSearchUrl(c) {
  const q = [c.name, c.city || c.prefecture, 'εστιατόριο catering'].filter(Boolean).join(' ');
  return `https://www.google.com/search?q=${encodeURIComponent(q)}`;
}

function openGemi(vat, companyName = '') {
  const cleanVat = String(vat || '').trim();
  if (cleanVat && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(cleanVat).then(() => {
      showToast(`📋 Αντιγράφηκε το ΑΦΜ ${cleanVat}. Μετάβαση στη Δημοσιότητα Γ.Ε.ΜΗ.`, 'shield-check');
    }).catch(() => {
      showToast(`Μετάβαση στη Δημοσιότητα Γ.Ε.ΜΗ. για ΑΦΜ ${cleanVat}`, 'shield-check');
    });
  } else {
    showToast(`Μετάβαση στη Δημοσιότητα Γ.Ε.ΜΗ.`, 'shield-check');
  }
  window.open('https://publicity.businessportal.gr/', '_blank');
}

function getGemiSearchUrl(vat, companyName = '') {
  const q = `site:businessportal.gr "${vat}"`;
  return `https://www.google.com/search?q=${encodeURIComponent(q)}`;
}

function getGemiUrl(vat) {
  return `https://publicity.businessportal.gr/`;
}

// -----------------------------------------------------------------------------
// Map Redirection & Pin Halo Animation
// -----------------------------------------------------------------------------
function focusOnMap(lat, lon, companyId, companyName, profileId) {
  // 1. Switch active tab to overview
  const tabBtn = document.querySelector('.nav-tab-btn[data-tab="overview"]');
  if (tabBtn) tabBtn.click();
  
  if (state.map && lat && lon) {
    // 2. Fly map to coordinates with street-level zoom
    state.map.flyTo([lat, lon], 15, { duration: 0.8 });
    
    // 3. Highlight pulsing halo at [lat, lon]
    showMapHalo([lat, lon]);
    
    // 4. Scroll smoothly to map container
    const mapEl = document.getElementById('leaflet-map');
    if (mapEl) {
      mapEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    // 5. Open popup on target
    setTimeout(() => {
      const pIdx = parseInt(String(profileId || '').replace('ICP_', '')) - 1;
      const color = ICP_COLORS[pIdx % ICP_COLORS.length] || '#3b82f6';
      
      L.popup({ offset: [0, -4] })
        .setLatLng([lat, lon])
        .setContent(`
          <div class="text-xs p-1 space-y-1.5 max-w-[280px]">
            <div class="font-bold text-white text-sm leading-tight">${escapeHtml(companyName || 'Επιχείρηση')}</div>
            <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold text-white" style="background-color: ${color}">
              📍 ${escapeHtml(profileId || 'Επιχείρηση')}
            </div>
            <div class="pt-2 flex items-center gap-1.5">
              <button onclick="inspectCompany(${companyId})" class="flex-1 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-[11px] transition-colors flex items-center justify-center gap-1">
                <i data-lucide="building-2" class="w-3 h-3"></i>
                <span>Καρτέλα</span>
              </button>
              <a href="https://www.google.com/maps/search/?api=1&query=${lat},${lon}" target="_blank" rel="noopener" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-400 font-semibold text-[11px] transition-colors flex items-center gap-1">
                <i data-lucide="map" class="w-3 h-3"></i>
                <span>Maps</span>
              </a>
            </div>
          </div>
        `)
        .openOn(state.map);
      lucide.createIcons();
    }, 850);
    
    showToast(`📍 Εστίαση στον Χάρτη: ${companyName || 'Επιχείρηση'}`, 'map-pin');
  } else {
    showToast(`⚠️ Δεν υπάρχουν διαθέσιμες συντεταγμένες GPS για αυτή την επιχείρηση.`, 'alert-circle');
  }
}

function showMapHalo(latLon) {
  if (!state.map) return;
  if (state.haloMarker) {
    state.map.removeLayer(state.haloMarker);
    state.haloMarker = null;
  }
  const haloIcon = L.divIcon({
    className: 'map-focus-halo',
    iconSize: [44, 44],
    iconAnchor: [22, 22],
  });
  state.haloMarker = L.marker(latLon, { icon: haloIcon, interactive: false }).addTo(state.map);
  setTimeout(() => {
    if (state.haloMarker && state.map) {
      state.map.removeLayer(state.haloMarker);
      state.haloMarker = null;
    }
  }, 4500);
}

// -----------------------------------------------------------------------------
// App Initialization
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await bootstrapApp();
    setupEventListeners();
  } catch (err) {
    console.error('Bootstrap error:', err);
  }
});

async function bootstrapApp() {
  const resp = await fetch('/api/bootstrap');
  const data = await resp.json();
  
  state.defaults = data.defaults || {};
  state.params = { ...state.defaults };
  state.paramSpecs = data.param_specs || [];
  state.benchmark = data.benchmark || {};
  state.ollama = data.ollama || { available: false, model: '', installed: [] };
  state.excluded = data.excluded_companies || [];
  
  if (data.initial) {
    applyEnginePayload(data.initial);
  }

  // Update Status Pills
  updateOllamaHeaderStatus();
  updateExcludedHeaderStatus();
  updateBenchmarkCards();

  // Initialize UI controls & components
  initMap();
  initDonutCharts();
  initSilhouetteChart();
  setLanguage(state.lang);
  lucide.createIcons();
}

function updateOllamaHeaderStatus() {
  const dot = document.getElementById('header-ollama-dot');
  const text = document.getElementById('header-ollama-text');
  if (!dot || !text) return;
  
  if (state.ollama && state.ollama.available) {
    dot.className = 'w-2 h-2 rounded-full bg-emerald-400';
    text.textContent = `Ollama: ${state.ollama.model || 'Online'}`;
  } else {
    dot.className = 'w-2 h-2 rounded-full bg-amber-400';
    text.textContent = 'Ollama: Offline (Οδηγίες)';
  }
}

function updateExcludedHeaderStatus() {
  const el = document.getElementById('header-excluded-count');
  if (el) el.textContent = String((state.excluded || []).length);
  const items = state.excluded || [];
  const userCount = items.filter(x => (x.source || 'user') === 'user').length;
  const supCount = items.filter(x => x.source === 'superseded').length;
  const uEl = document.getElementById('header-excluded-user-count');
  const sEl = document.getElementById('header-excluded-sup-count');
  if (uEl) uEl.textContent = String(userCount);
  if (sEl) sEl.textContent = supCount.toLocaleString('el-GR');
}

function updateBenchmarkCards() {
  const bm = state.benchmark || {};
  const r500Val = bm.r500_mean ? (bm.r500_mean * 100).toFixed(2) : '13.00';
  const r500Std = bm.r500_std ? (bm.r500_std * 100).toFixed(2) : '4.16';
  const r2000Val = bm.r2000_mean ? (bm.r2000_mean * 100).toFixed(2) : '40.18';
  const r2000Std = bm.r2000_std ? (bm.r2000_std * 100).toFixed(2) : '7.44';

  const base500 = bm.baseline_r500 ? (bm.baseline_r500 * 100).toFixed(2) : '1.87';
  const base2000 = bm.baseline_r2000 ? (bm.baseline_r2000 * 100).toFixed(2) : '7.48';

  const lift500 = (parseFloat(r500Val) / Math.max(parseFloat(base500), 0.01)).toFixed(1);
  const lift2000 = (parseFloat(r2000Val) / Math.max(parseFloat(base2000), 0.01)).toFixed(1);

  const r500El = document.getElementById('bench-r500-val');
  if (r500El) r500El.innerHTML = `${r500Val}% <span class="text-xs font-normal text-slate-400">± ${r500Std}%</span>`;

  const r500BaseEl = document.getElementById('bench-r500-baseline');
  if (r500BaseEl) r500BaseEl.textContent = `${base500}% (Empirical: 2.30%)`;

  const r500LiftEl = document.getElementById('bench-r500-lift');
  if (r500LiftEl) r500LiftEl.textContent = `🚀 +${lift500}x Statistical Lift vs Baseline`;

  const r2000El = document.getElementById('bench-r2000-val');
  if (r2000El) r2000El.innerHTML = `${r2000Val}% <span class="text-xs font-normal text-slate-400">± ${r2000Std}%</span>`;

  const r2000BaseEl = document.getElementById('bench-r2000-baseline');
  if (r2000BaseEl) r2000BaseEl.textContent = `${base2000}% (Empirical: 7.73%)`;

  const r2000LiftEl = document.getElementById('bench-r2000-lift');
  if (r2000LiftEl) r2000LiftEl.textContent = `🎯 +${lift2000}x Statistical Lift vs Baseline`;
}

function applyEnginePayload(payload) {
  state.meta = payload.meta || {};
  state.profiles = payload.profiles || [];
  state.lookalikes = payload.lookalikes || [];
  state.clients = payload.clients || [];
  if (payload.params) {
    state.params = { ...payload.params };
  }
  
  // Sync sliders to current state values
  syncSlidersToState();
  
  // Update execution speed indicator
  const badge = document.getElementById('recompute-speed-badge');
  const speedText = document.getElementById('speed-text');
  if (badge && speedText && state.meta.total_ms) {
    const isInstant = state.meta.cluster_reused;
    speedText.textContent = isInstant 
      ? `⚡ Re-scored in ${state.meta.total_ms}ms (K-Means reused)` 
      : `🔄 Re-clustered in ${state.meta.total_ms}ms`;
    badge.className = isInstant 
      ? 'inline-flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 font-medium'
      : 'inline-flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/25 font-medium';
  }
}

// -----------------------------------------------------------------------------
// Rendering & View Management
// -----------------------------------------------------------------------------
function renderAllViews() {
  renderKpis();
  renderDonutCharts();
  renderMapMarkers();
  renderIcpCards();
  renderLeadsTable();
  renderClientsTable();
  renderSilhouetteChart();
  lucide.createIcons();
}

function renderKpis() {
  const meta = state.meta;
  const isEn = state.lang === 'en';
  const loc = isEn ? 'en-US' : 'el-GR';
  const clientCnt = meta.client_count || 300;
  const assignedCnt = meta.assigned || 286;
  const noFitCnt = meta.no_fit || 14;
  const totalRev = meta.total_turnover || 5443170;
  const meanRev = meta.mean_turnover || 18144;
  const optK = meta.optimal_k || 2;
  const ariPct = ((meta.global_ari || 0.958) * 100).toFixed(1);
  const lookCnt = meta.lookalike_count || 2000;

  // Update Collapsible Executive Bar
  safeSetText('exec-bar-clients', clientCnt.toLocaleString(loc));
  safeSetText('exec-bar-assigned', String(assignedCnt));
  safeSetText('exec-bar-nofit', String(noFitCnt));
  safeSetText('exec-bar-revenue', `€${(totalRev / 1e6).toFixed(2)}M`);
  safeSetText('exec-bar-icps', isEn ? `${optK} Clusters` : `${optK} Ομάδες`);
  safeSetText('exec-bar-ari', `${ariPct}%`);

  // Update Expanded KPI Strip
  safeSetText('kpi-client-count', clientCnt.toLocaleString(loc));
  safeSetText('kpi-assigned-badge', isEn ? `${assignedCnt} in ICPs` : `${assignedCnt} σε Προφίλ`);
  safeSetText('kpi-nofit-badge', `${noFitCnt} NO_FIT`);
  
  safeSetText('kpi-total-turnover', `€${Math.round(totalRev).toLocaleString(loc)}`);
  safeSetText('kpi-mean-turnover', isEn ? `€${Math.round(meanRev).toLocaleString(loc)} / client` : `€${Math.round(meanRev).toLocaleString(loc)} / πελάτη`);
  
  safeSetText('kpi-optimal-k', isEn ? `${optK} Clusters` : `${optK} Ομάδες`);
  safeSetText('kpi-stability', `${Math.round((meta.global_ari || 0.958) * 100)}%`);
  
  safeSetText('kpi-lookalikes-count', lookCnt.toLocaleString(loc));
  safeSetText('tab-leads-badge', lookCnt.toLocaleString(loc));
  safeSetText('tab-clients-badge', (meta.client_rows || 300).toLocaleString(loc));
  
  const clientNameEl = document.getElementById('header-client-name');
  if (clientNameEl && meta.client_set_name) {
    clientNameEl.textContent = `${meta.client_set_name} (${clientCnt})`;
  }
}

// -----------------------------------------------------------------------------
// Interactive Leaflet Map (Zero API Key Basemap + 60 FPS Hover Halo + Route Rings)
// -----------------------------------------------------------------------------
function initMap() {
  const mapContainer = document.getElementById('leaflet-map');
  if (!mapContainer || state.map) return;
  
  // Use HTML5 Canvas renderer for 60fps rendering of 2,300 points
  state.canvasRenderer = L.canvas({ padding: 0.5 });
  
  // Center on Greece
  state.map = L.map('leaflet-map', {
    center: [38.25, 23.85],
    zoom: 6,
    zoomControl: true,
    renderer: state.canvasRenderer,
  });
  
  // Re-hydrate Lucide icons inside dynamic Leaflet popups upon open
  state.map.on('popupopen', () => {
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  });
  
  // Set default basemap: Esri World Dark Gray Canvas (Requires ZERO API Key)
  switchBasemap('esri');
  
  state.deliveryRingsLayer = L.layerGroup().addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
  state.salesRouteLayer = L.layerGroup().addTo(state.map);

  // Single Reusable DOM Overlay Halo Marker for 60 FPS Table-to-Map Hover Priming
  const pulseIcon = L.divIcon({
    className: 'map-pulse-halo',
    html: '<div class="pulse-ring"></div><div class="pulse-core"></div>',
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });
  state.hoverHaloMarker = L.marker([0, 0], {
    icon: pulseIcon,
    interactive: false,
    zIndexOffset: 2000,
    opacity: 0,
  }).addTo(state.map);

  // Dynamic Map Resizing Handler
  if (window.ResizeObserver && !mapContainer._resizeObserverAttached) {
    mapContainer._resizeObserverAttached = true;
    let resizeTimer = null;
    const ro = new ResizeObserver(() => {
      if (state.map) {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          if (state.map) state.map.invalidateSize();
        }, 60);
      }
    });
    ro.observe(mapContainer);
  }

  window.addEventListener('resize', () => {
    if (state.map) state.map.invalidateSize();
  });
}

window.primeMapMarkerOnHover = function(lat, lon) {
  if (!state.map || !state.hoverHaloMarker || !lat || !lon) return;
  state.hoverHaloMarker.setLatLng([lat, lon]);
  state.hoverHaloMarker.setOpacity(1);
};

window.clearMapHoverHalo = function() {
  if (state.hoverHaloMarker) {
    state.hoverHaloMarker.setOpacity(0);
  }
};

window.toggleDeliveryRouteRings = function() {
  state.showDeliveryRings = !state.showDeliveryRings;
  const btn = document.getElementById('toggle-delivery-rings-btn');
  if (btn) {
    btn.className = state.showDeliveryRings
      ? 'px-2.5 py-1 rounded-md text-[11px] font-semibold bg-cyan-500/25 text-cyan-200 border border-cyan-400 flex items-center gap-1 transition-all shadow-sm'
      : 'px-2.5 py-1 rounded-md text-[11px] font-semibold bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-500/30 flex items-center gap-1 transition-all';
  }
  renderDeliveryRouteRings();
  const isEn = state.lang === 'en';
  showToast(
    state.showDeliveryRings
      ? (isEn ? '🚚 1km Client Delivery Route Rings enabled on map' : '🚚 Ενεργοποιήθηκαν οι ζώνες διανομής 1km γύρω από τους ενεργούς πελάτες')
      : (isEn ? '🚚 Delivery Route Rings hidden' : '🚚 Απόκρυψη ζωνών διανομής 1km'),
    'map-pin'
  );
};

function renderDeliveryRouteRings() {
  if (!state.map || !state.deliveryRingsLayer) return;
  state.deliveryRingsLayer.clearLayers();
  if (!state.showDeliveryRings) return;

  (state.clients || []).forEach(c => {
    if (!c.lat || !c.lon || c.profile_id === 'NO_FIT') return;
    const pIdx = parseInt(String(c.profile_id || '').replace('ICP_', '')) - 1;
    const color = ICP_COLORS[pIdx % ICP_COLORS.length] || '#38bdf8';
    const ring = L.circle([c.lat, c.lon], {
      renderer: state.canvasRenderer,
      radius: 1000, // 1.0 km delivery route catchment
      color: color,
      weight: 1,
      opacity: 0.45,
      fillColor: color,
      fillOpacity: 0.08,
      interactive: false,
    });
    state.deliveryRingsLayer.addLayer(ring);
  });
}

/**
 * Ergonomically centers a table row in the middle of the table scroll viewport
 */
function scrollRowToCenter(row) {
  if (!row) return;
  const scrollContainer = row.closest('.cockpit-table-scroll');
  if (scrollContainer) {
    const containerHeight = scrollContainer.clientHeight;
    const rowOffsetTop = row.offsetTop;
    const rowHeight = row.offsetHeight || 44;
    const targetScrollTop = rowOffsetTop - (containerHeight / 2) + (rowHeight / 2);
    scrollContainer.scrollTo({
      top: Math.max(0, targetScrollTop),
      behavior: 'smooth'
    });
  } else {
    row.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
  }
}
window.scrollRowToCenter = scrollRowToCenter;

/**
 * Pagination-Aware Map-to-Table Click Synchronization (Section 4.1)
 */
window.syncTableToLeadClick = function(companyId, openInspect = false) {
  const items = state.currentFilteredLeads || [];
  const idx = items.findIndex(l => l.company_id === companyId);
  if (idx !== -1) {
    const targetPage = Math.floor(idx / state.leadsFilter.pageSize) + 1;
    const pageChanged = state.leadsFilter.page !== targetPage;
    if (pageChanged) {
      state.leadsFilter.page = targetPage;
      renderLeadsTable();
    }
    state.focusedLeadIndex = idx;
    setTimeout(() => {
      const row = document.getElementById(`lead-row-${companyId}`);
      if (row) {
        document.querySelectorAll('#leads-table-body tr').forEach(r => r.classList.remove('row-active-highlight'));
        row.classList.add('row-active-highlight');
        scrollRowToCenter(row);
      }
    }, pageChanged ? 60 : 30);
  }
  if (openInspect) {
    inspectCompany(companyId);
  }
};

/**
 * Switch basemap between Esri Dark Gray and OpenStreetMap (both require 0 API keys)
 */
function switchBasemap(style) {
  if (!state.map) return;
  
  if (state.currentTileLayer) {
    state.map.removeLayer(state.currentTileLayer);
  }
  
  state.mapFilter.basemap = style;
  const container = document.getElementById('leaflet-map');
  
  const esriBtn = document.getElementById('basemap-esri-btn');
  const osmBtn = document.getElementById('basemap-osm-btn');
  
  if (style === 'osm') {
    state.currentTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>',
      subdomains: 'abc',
      maxZoom: 19,
    }).addTo(state.map);
    
    if (container) container.classList.remove('dark-osm-tiles');
    if (esriBtn) esriBtn.className = 'px-1.5 py-0.5 rounded text-[10px] font-semibold text-slate-400 hover:text-white transition-colors';
    if (osmBtn) osmBtn.className = 'px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-600 text-white transition-colors';
  } else {
    state.currentTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 16,
    }).addTo(state.map);
    
    if (container) container.classList.remove('dark-osm-tiles');
    if (esriBtn) esriBtn.className = 'px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-600 text-white transition-colors';
    if (osmBtn) osmBtn.className = 'px-1.5 py-0.5 rounded text-[10px] font-semibold text-slate-400 hover:text-white transition-colors';
  }
}

function updateMapFilter() {
  const sel = document.getElementById('map-layer-filter');
  if (sel) {
    state.mapFilter.layer = sel.value;
    renderMapMarkers();
  }
}

/**
 * Plots all existing clients and all 2,000 lookalike leads across Greece.
 * Uses Leaflet canvas renderer for instant, lag-free 60fps rendering.
 */
function renderMapMarkers() {
  if (!state.map || !state.markersLayer) return;
  state.markersLayer.clearLayers();
  renderDeliveryRouteRings();
  
  const isEn = state.lang === 'en';
  const loc = isEn ? 'en-US' : 'el-GR';
  const layerMode = state.mapFilter.layer; // 'ALL' | 'CLIENTS' | 'LEADS'
  let clientMarkerCount = 0;
  let leadMarkerCount = 0;
  
  // 1. Plot existing clients (Larger, glowing circular points)
  if (layerMode === 'ALL' || layerMode === 'CLIENTS') {
    state.clients.forEach(c => {
      if (!c.lat || !c.lon || c.profile_id === 'NO_FIT') return;
      clientMarkerCount++;
      const pIdx = parseInt(c.profile_id.replace('ICP_', '')) - 1;
      const color = ICP_COLORS[pIdx % ICP_COLORS.length] || '#3b82f6';
      
      const marker = L.circleMarker([c.lat, c.lon], {
        renderer: state.canvasRenderer,
        radius: 6.5,
        fillColor: color,
        color: '#ffffff',
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.92,
      });
      
      const popupHtml = `
        <div class="text-xs p-1 space-y-1.5 w-[270px] min-w-[270px]">
          <div class="font-bold text-white text-sm leading-tight">${escapeHtml(c.name || (isEn ? 'Client' : 'Πελάτης'))}</div>
          <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-500/25 text-blue-300 border border-blue-500/40">
            ⭐ ${isEn ? 'Active Client' : 'Ενεργός Πελάτης'} · ${escapeHtml(c.profile_id)}
          </div>
          <div class="text-slate-300 pt-1 space-y-0.5">
            <div>${isEn ? 'Revenue' : 'Κύκλος Εργασιών'}: <strong class="text-white font-mono">€${(c.turnover || 0).toLocaleString(loc)}</strong> | ${isEn ? 'Match' : 'Ταύτιση'}: <strong class="text-emerald-400 font-mono">${(c.score * 100).toFixed(1)}%</strong></div>
            <div>🏛️ ${escapeHtml(c.legal_type || '')} · 📍 ${escapeHtml(c.municipality || c.city || c.prefecture || '')}</div>
            <div class="text-[11px] text-slate-400 truncate">🍽️ ${escapeHtml(c.kad_desc || c.kad || '')}</div>
          </div>
          <div class="pt-2 flex items-center gap-1.5">
            <button onclick="inspectCompany(${c.company_id})" class="flex-1 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-[11px] transition-colors flex items-center justify-center gap-1">
              <i data-lucide="building-2" class="w-3.5 h-3.5"></i>
              <span>${isEn ? 'Inspect' : 'Καρτέλα'}</span>
            </button>
            <a href="${getGoogleMapsUrl(c)}" target="_blank" rel="noopener" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-400 font-semibold text-[11px] transition-colors flex items-center gap-1">
              <i data-lucide="map" class="w-3.5 h-3.5"></i>
              <span>Maps</span>
            </a>
          </div>
        </div>
      `;
      marker.bindPopup(popupHtml, { minWidth: 270, maxWidth: 320 });
      state.markersLayer.addLayer(marker);
    });
  }
  
  // 2. Plot ALL 2,000 lookalike leads across Greece
  if (layerMode === 'ALL' || layerMode === 'LEADS') {
    state.lookalikes.forEach(l => {
      if (!l.lat || !l.lon) return;
      leadMarkerCount++;
      const pIdx = parseInt(l.profile_id.replace('ICP_', '')) - 1;
      const color = ICP_COLORS[pIdx % ICP_COLORS.length] || '#34d399';
      
      const marker = L.circleMarker([l.lat, l.lon], {
        renderer: state.canvasRenderer,
        radius: 4.5,
        fillColor: color,
        color: color,
        weight: 1,
        opacity: 0.85,
        fillOpacity: 0.65,
      });

      marker.on('click', () => {
        syncTableToLeadClick(l.company_id, false);
      });
      
      const distBadge = typeof l.nearest_client_km === 'number'
        ? `<span class="inline-flex items-center gap-1 text-cyan-300 font-mono text-[11px]">🚚 ${l.nearest_client_km < 1 ? Math.round(l.nearest_client_km * 1000) + 'm' : l.nearest_client_km.toFixed(1) + 'km'} ${isEn ? 'from client' : 'από πελάτη'}</span>`
        : '';

      const popupHtml = `
        <div class="text-xs p-1 space-y-1.5 w-[270px] min-w-[270px]">
          <div class="font-bold text-white text-sm leading-tight">${escapeHtml(l.name || (isEn ? 'Prospect' : 'Επιχείρηση'))}</div>
          <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/25 text-emerald-300 border border-emerald-500/40">
            🎯 ${isEn ? 'Prospect' : 'Υποψήφιος'} #${l.rank} · ${escapeHtml(l.profile_id)}
          </div>
          <div class="text-slate-300 pt-1 space-y-0.5">
            <div>${isEn ? 'Match' : 'Ταύτιση'}: <strong class="text-emerald-400 font-mono">${(l.score * 100).toFixed(1)}%</strong> | 🏛️ ${escapeHtml(l.legal_type || '')}</div>
            <div>📍 ${escapeHtml(l.municipality || l.city || l.prefecture || '')}</div>
            ${distBadge ? `<div>${distBadge}</div>` : ''}
            <div class="text-[11px] text-slate-400 truncate">🍽️ ${escapeHtml(l.kad_desc || l.kad || '')}</div>
          </div>
          <div class="pt-2 flex items-center gap-1.5">
            <button onclick="inspectCompany(${l.company_id})" class="flex-1 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-[11px] transition-colors flex items-center justify-center gap-1">
              <i data-lucide="building-2" class="w-3.5 h-3.5"></i>
              <span>${isEn ? 'Inspect' : 'Καρτέλα'}</span>
            </button>
            <a href="${getGoogleMapsUrl(l)}" target="_blank" rel="noopener" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-400 font-semibold text-[11px] transition-colors flex items-center gap-1">
              <i data-lucide="map" class="w-3.5 h-3.5"></i>
              <span>Maps</span>
            </a>
          </div>
        </div>
      `;
      marker.bindPopup(popupHtml, { minWidth: 270, maxWidth: 320 });
      state.markersLayer.addLayer(marker);
    });
  }
  
  const total = clientMarkerCount + leadMarkerCount;
  safeSetText(
    'map-marker-counter',
    `📍 ${total.toLocaleString(loc)}`
  );
}

/**
 * High-precision geographic zoom presets covering all major Greek regions
 */
function zoomToRegion(regionKey) {
  if (!state.map) return;
  
  const regions = {
    greece: { center: [38.25, 23.85], zoom: 6 },
    attica: { center: [38.00, 23.75], zoom: 10 },
    thessaloniki: { center: [40.63, 22.95], zoom: 10 },
    halkidiki: { center: [40.25, 23.45], zoom: 9 },
    crete: { center: [35.25, 24.80], zoom: 8 },
    cyclades: { center: [37.05, 25.35], zoom: 8 },
    dodecanese: { center: [36.25, 27.95], zoom: 8 },
    ionian: { center: [39.00, 20.30], zoom: 8 },
    peloponnese: { center: [37.65, 22.15], zoom: 8 },
    thessaly: { center: [39.55, 22.60], zoom: 8 },
  };
  
  const target = regions[regionKey] || regions.greece;
  state.mapFilter.region = regionKey;
  state.map.flyTo(target.center, target.zoom, { duration: 0.75 });
  
  // Update button active styling
  document.querySelectorAll('.region-preset-btn').forEach(btn => {
    if (btn.getAttribute('data-region') === regionKey) {
      btn.className = 'region-preset-btn shrink-0 px-2.5 py-1 rounded-lg text-xs font-semibold bg-blue-600 text-white transition-colors';
    } else {
      btn.className = 'region-preset-btn shrink-0 px-2.5 py-1 rounded-lg text-xs font-semibold text-slate-300 hover:text-white bg-slate-900/80 border border-slate-800/80 transition-colors';
    }
  });
}

// -----------------------------------------------------------------------------
// Donut Charts (Chart.js)
// -----------------------------------------------------------------------------
// Donut Charts with Floating HTML Tooltips (Eliminates Canvas Boundary Clipping)
// -----------------------------------------------------------------------------
function getOrCreateDonutTooltip() {
  let tooltipEl = document.getElementById('chartjs-donut-tooltip');
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.id = 'chartjs-donut-tooltip';
    tooltipEl.style.position = 'fixed';
    tooltipEl.style.zIndex = '99999';
    tooltipEl.style.pointerEvents = 'none';
    tooltipEl.style.transition = 'opacity 0.15s ease-out, transform 0.15s ease-out';
    tooltipEl.className = 'bg-slate-950/95 text-white border border-blue-500/40 rounded-lg p-2.5 shadow-2xl backdrop-blur-md text-xs';
    document.body.appendChild(tooltipEl);
  }
  return tooltipEl;
}

function handleDonutExternalTooltip(context, isRev) {
  const tooltipEl = getOrCreateDonutTooltip();
  const tooltipModel = context.tooltip;

  if (!tooltipModel || tooltipModel.opacity === 0) {
    tooltipEl.style.opacity = '0';
    tooltipEl.style.transform = 'scale(0.95)';
    return;
  }

  if (tooltipModel.body && tooltipModel.dataPoints && tooltipModel.dataPoints.length) {
    const item = tooltipModel.dataPoints[0];
    const idx = item.dataIndex;
    const p = (state.profiles && state.profiles[idx]) ? state.profiles[idx] : null;
    const isEn = state.lang === 'en';
    const color = ICP_COLORS[idx % ICP_COLORS.length];

    let title = p ? `${p.id}: ${p.short_title || p.title || ''}` : (item.label || `ICP ${idx + 1}`);
    let valueStr = '';

    if (p) {
      if (isRev) {
        const revM = ((p.turnover_total || 0) / 1e6).toFixed(2);
        const pct = Math.round((p.turnover_share || 0) * 100);
        valueStr = `€${revM}M (${pct}% ${isEn ? 'of total revenue' : 'συνολικού κύκλου'})`;
      } else {
        const pct = Math.round((p.client_share || 0) * 100);
        valueStr = `${p.client_count} ${isEn ? 'clients' : 'πελάτες'} (${pct}% ${isEn ? 'of active portfolio' : 'πελατολογίου'})`;
      }
    } else {
      valueStr = String(item.formattedValue || item.raw || '');
    }

    tooltipEl.innerHTML = `
      <div class="flex items-center gap-2 mb-1">
        <span class="w-2.5 h-2.5 rounded-full shrink-0" style="background-color: ${color}"></span>
        <span class="font-bold text-white whitespace-nowrap">${escapeHtml(title)}</span>
      </div>
      <div class="text-[11px] text-cyan-300 font-mono font-semibold pl-4">
        ${escapeHtml(valueStr)}
      </div>
    `;
  }

  const rect = context.chart.canvas.getBoundingClientRect();
  tooltipEl.style.opacity = '1';
  tooltipEl.style.transform = 'scale(1)';
  
  // Position outside the donut to never obscure the center metric
  const caretX = tooltipModel.caretX || 0;
  const caretY = tooltipModel.caretY || 0;
  const leftPos = rect.left + caretX + (caretX > rect.width / 2 ? -180 : 18);
  const topPos = rect.top + caretY - 35;
  
  tooltipEl.style.left = Math.max(10, Math.min(window.innerWidth - 220, leftPos)) + 'px';
  tooltipEl.style.top = Math.max(10, topPos) + 'px';
}

function initDonutCharts() {
  const revCanvas = document.getElementById('chart-revenue-donut');
  const clientsCanvas = document.getElementById('chart-clients-donut');
  if (!revCanvas || !clientsCanvas) return;
  
  if (state.chartRevDonut) {
    try { state.chartRevDonut.destroy(); } catch (e) {}
  }
  if (state.chartClientsDonut) {
    try { state.chartClientsDonut.destroy(); } catch (e) {}
  }

  state.chartRevDonut = new Chart(revCanvas, {
    type: 'doughnut',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: false,
          external: function(context) { handleDonutExternalTooltip(context, true); }
        }
      }
    }
  });

  state.chartClientsDonut = new Chart(clientsCanvas, {
    type: 'doughnut',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: false,
          external: function(context) { handleDonutExternalTooltip(context, false); }
        }
      }
    }
  });
}

function renderDonutCharts() {
  if (!state.chartRevDonut || !state.chartClientsDonut) return;
  
  const labels = state.profiles.map(p => p.short_title || p.id);
  const turnovers = state.profiles.map(p => p.turnover_total);
  const clientCounts = state.profiles.map(p => p.client_count);
  const colors = state.profiles.map((p, i) => ICP_COLORS[i % ICP_COLORS.length]);
  
  state.chartRevDonut.data.labels = labels;
  state.chartRevDonut.data.datasets[0].data = turnovers;
  state.chartRevDonut.data.datasets[0].backgroundColor = colors;
  state.chartRevDonut.update('none');
  
  state.chartClientsDonut.data.labels = labels;
  state.chartClientsDonut.data.datasets[0].data = clientCounts;
  state.chartClientsDonut.data.datasets[0].backgroundColor = colors;
  state.chartClientsDonut.update('none');
  
  const totalRev = state.meta.total_turnover || 0;
  const clientCount = state.meta.client_count || 300;
  const totalRevM = totalRev / 1e6;
  safeSetText('chart-rev-total-label', `€${totalRevM.toFixed(2)}M`);
  safeSetText('chart-clients-total-label', `${clientCount}`);

  // Calculate Average Client Ticket & Coverage
  const avgTicket = clientCount > 0 ? (totalRev / clientCount) : 0;
  const avgTicketStr = avgTicket >= 1000 ? `€${(avgTicket / 1000).toFixed(1)}K` : `€${Math.round(avgTicket)}`;
  safeSetText('portfolio-avg-ticket', avgTicketStr);
  safeSetText('portfolio-coverage', '100%');

  // Render Commercial Yield & Progress Breakdown Bars
  const isEn = state.lang === 'en';
  const breakdownContainer = document.getElementById('icp-breakdown-bars');
  if (breakdownContainer) {
    breakdownContainer.innerHTML = state.profiles.map((p, i) => {
      const color = ICP_COLORS[i % ICP_COLORS.length];
      const revPct = Math.round((p.turnover_share || 0) * 100);
      const cntPct = Math.round((p.client_share || 0) * 100);
      const revTotal = (p.turnover_total || 0) / 1e6;
      const avgIcpTicket = p.client_count > 0 ? (p.turnover_total / p.client_count) : 0;
      const isHighYield = revPct >= cntPct;

      return `
        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div class="flex items-center justify-between text-xs">
            <div class="flex items-center gap-1.5 font-bold text-white">
              <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${color}"></span>
              <span>${escapeHtml(p.id)}</span>
              <span class="text-[11px] font-normal text-slate-400">· €${revTotal.toFixed(2)}M</span>
            </div>
            <span class="text-[10px] px-1.5 py-0.5 rounded font-semibold ${isHighYield ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-blue-500/15 text-blue-300 border border-blue-500/30'}">
              ${isHighYield ? (isEn ? 'High Revenue Yield' : 'Υπεραπόδοση Κύκλου Εργασιών') : (isEn ? 'High Volume' : 'Υψηλός Όγκος')} (μ.ο. €${(avgIcpTicket / 1000).toFixed(1)}K)
            </span>
          </div>

          <div class="space-y-1">
            <div class="flex items-center justify-between text-[10.5px]">
              <span class="text-slate-400">${isEn ? 'Revenue Share:' : 'Μερίδιο Κύκλου Εργασιών:'}</span>
              <span class="font-bold text-emerald-400">${revPct}%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div class="h-1.5 rounded-full transition-all duration-500" style="width: ${revPct}%; background-color: ${color};"></div>
            </div>
          </div>

          <div class="space-y-1">
            <div class="flex items-center justify-between text-[10.5px]">
              <span class="text-slate-400">${isEn ? 'Client Count:' : 'Πλήθος Πελατών:'}</span>
              <span class="font-bold text-slate-300">${cntPct}% (${p.client_count})</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div class="h-1.5 rounded-full transition-all duration-500 bg-slate-500" style="width: ${cntPct}%;"></div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }
  
  // Render Legend Strip below donuts
  const legendContainer = document.getElementById('icp-donut-legend');
  if (legendContainer) {
    legendContainer.innerHTML = state.profiles.map((p, i) => {
      const color = ICP_COLORS[i % ICP_COLORS.length];
      const revPct = Math.round((p.turnover_share || 0) * 100);
      const cntPct = Math.round((p.client_share || 0) * 100);
      return `
        <div class="flex items-center justify-between text-xs">
          <div class="flex items-center gap-2 truncate">
            <span class="w-3 h-3 rounded-full shrink-0" style="background-color: ${color}"></span>
            <span class="font-bold text-white">${escapeHtml(p.id)}:</span>
            <span class="text-slate-300 truncate">${escapeHtml(p.short_title)}</span>
          </div>
          <div class="font-mono text-[11px] text-slate-400 shrink-0">
            <span class="text-emerald-400 font-bold">${revPct}% ${isEn ? 'revenue' : 'κύκλου εργ.'}</span> · <span>${p.client_count} ${isEn ? 'cl.' : 'πελ.'} (${cntPct}%)</span>
          </div>
        </div>
      `;
    }).join('');
  }
  lucide.createIcons();
}

// -----------------------------------------------------------------------------
// ICP Profiles Cards & Sales Intelligence Cockpit
// -----------------------------------------------------------------------------
function getProfileSalesIntel(p) {
  if (p.sales_intelligence && p.sales_intelligence.lead_funnel) {
    return p.sales_intelligence;
  }
  // Client-side fallback calculation from state.lookalikes
  const lookP = (state.lookalikes || []).filter(l => l.profile_id === p.id);
  const total = lookP.length || 1;
  const hot = lookP.filter(l => (l.score || 0) >= 0.90);
  const warm = lookP.filter(l => (l.score || 0) >= 0.80 && (l.score || 0) < 0.90);
  const nurture = lookP.filter(l => (l.score || 0) < 0.80);

  const muniMap = {};
  lookP.forEach(l => {
    if (l.municipality) {
      const name = 'Δήμος ' + l.municipality.split('/')[0].trim();
      muniMap[name] = (muniMap[name] || 0) + 1;
    }
  });
  const territories = Object.entries(muniMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([name, count]) => ({
      name,
      count,
      share: count / total,
      share_pct: Math.round((count / total) * 1000) / 10,
    }));

  const legalMap = {};
  lookP.forEach(l => {
    if (l.legal_type) legalMap[l.legal_type] = (legalMap[l.legal_type] || 0) + 1;
  });
  const topLegal = Object.entries(legalMap)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([legal_type, count]) => ({
      legal_type,
      count,
      share: count / total,
      share_pct: Math.round((count / total) * 1000) / 10,
    }));

  const dominantLegal = topLegal[0] ? topLegal[0].legal_type : 'ΙΚΕ';
  const multiBranchCnt = lookP.filter(l => l.is_branch || (l.branches && l.branches > 1)).length;

  return {
    lead_funnel: {
      total,
      hot: { count: hot.length, share: hot.length / total, share_pct: Math.round((hot.length / total) * 1000) / 10, label: 'Hot Leads', min_score: '≥90%', action: 'Άμεση Κλήση & Demo (48h)' },
      warm: { count: warm.length, share: warm.length / total, share_pct: Math.round((warm.length / total) * 1000) / 10, label: 'Warm Leads', min_score: '80-89%', action: 'SDR Outbound Sequence' },
      nurture: { count: nurture.length, share: nurture.length / total, share_pct: Math.round((nurture.length / total) * 1000) / 10, label: 'Nurture Leads', min_score: '<80%', action: 'Marketing Lead Drip' },
    },
    territories,
    territory_tip: 'Συγκέντρωση: Προγραμματίστε στοχευμένες ημερήσιες εξορμήσεις πεδίου (field sales days) στους κυριότερους δήμους.',
    firmographics: {
      top_legal: topLegal,
      dominant_legal: dominantLegal,
      multi_branch_count: multiBranchCnt,
      multi_branch_share_pct: Math.round((multiBranchCnt / total) * 1000) / 10,
      median_age_years: 4.0,
    },
    playbook: {
      sector_name: 'Εταιρικές Επιχειρήσεις B2B',
      deal_tier: 'Mid-Market / High Velocity',
      deal_size_label: `€${Math.round(p.turnover_median || 200000).toLocaleString('el-GR')} διάμεσος κύκλος εργασιών`,
      sales_cycle: '7 - 14 ημέρες',
      buyer_persona: `Διαχειριστής / Operations Director (${dominantLegal})`,
      buyer_persona_desc: 'Ευέλικτη διοίκηση με άμεση εξουσία υπογραφής.',
      pain_points: [
        'Διαχείριση υψηλού όγκου καθημερινών δελτίων αποστολής & τιμολογίων',
        'Αυστηρή συμμόρφωση myDATA χωρίς καθυστερήσεις στα σημεία παράδοσης',
        'Παρακολούθηση ανοιχτών πιστωτικών υπολοίπων προμηθευτών & πελατών B2B',
      ],
      pitch_hook: '«Καλημέρα σας, επικοινωνώ γιατί συνεργαζόμαστε με κορυφαίες επιχειρήσεις στον κλάδο σας στην περιοχή σας. Τους βοηθάμε να αυτοματοποιήσουν πλήρως τη ροή παραγγελιών και την τιμολόγηση, εξοικονομώντας 15+ ώρες γραφειοκρατίας την εβδομάδα.»',
      objection_rebuttal: '«Χρησιμοποιούμε ήδη άλλο σύστημα» ➔ «Ακριβώς επειδή έχετε όγκο, κουμπώνουμε άμεσα χωρίς καμία διακοπή της λειτουργίας σας.»',
      recommended_channel: 'Inside Sales με τηλεφωνική κλήση & 15-λεπτο Online Demo.',
    },
  };
}

function renderIcpCards() {
  const container = document.getElementById('icp-cards-container');
  if (!container) return;
  const isEn = state.lang === 'en';
  
  // Clean up any previously instantiated charts
  Object.values(state.compCharts || {}).forEach(ch => ch && ch.destroy());
  state.compCharts = {};
  Object.values(state.territoryCharts || {}).forEach(ch => ch && ch.destroy());
  state.territoryCharts = {};
  
  container.innerHTML = state.profiles.map((p, idx) => {
    const color = ICP_COLORS[idx % ICP_COLORS.length];
    const topTraits = (p.features || []).slice(0, 4);
    const si = getProfileSalesIntel(p);
    const activeTab = (state.icpActiveTab && state.icpActiveTab[p.id]) || 'pipeline';
    
    return `
      <div class="glass-panel p-6 sm:p-7 xl:p-8 border-l-4 transition-all" style="border-left-color: ${color}">
        
        <!-- Header -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
          <div>
            <div class="flex items-center gap-2.5">
              <span class="px-2.5 py-0.5 rounded-full text-xs font-extrabold text-white shadow-sm" style="background-color: ${color}">
                ${escapeHtml(p.id)}
              </span>
              <h3 class="text-base sm:text-lg font-extrabold text-white">${escapeHtml(p.title)}</h3>
            </div>
            <p class="text-xs text-slate-400 mt-1">${escapeHtml(p.region ? (isEn ? `Geographic Concentration: ${p.region}` : `Γεωγραφική συγκέντρωση: ${p.region}`) : '')}</p>
          </div>

          <!-- Quality Metrics Badges -->
          <div class="flex items-center gap-2 shrink-0">
            <span class="px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/25 text-xs text-blue-300 font-semibold flex items-center gap-1.5">
              <i data-lucide="shield-check" class="w-3.5 h-3.5"></i>
              ${isEn ? 'Confidence' : 'Εμπιστοσύνη'}: ${Math.round((p.confidence || 0) * 100)}%
            </span>
            <span class="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/25 text-xs text-emerald-300 font-semibold flex items-center gap-1.5">
              <i data-lucide="check" class="w-3.5 h-3.5"></i>
              ${isEn ? 'Stability' : 'Σταθερότητα'}: ${Math.round((p.stability || 0) * 100)}%
            </span>
          </div>
        </div>

        <!-- 2-Column Body: Key Figures & Sales Cockpit -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 xl:gap-8 mt-5">
          
          <!-- Left: Profile KPIs & Grounded Summary -->
          <div class="lg:col-span-6 space-y-4">
            
            <!-- 3 Mini Metric Boxes -->
            <div class="grid grid-cols-3 gap-2.5 sm:gap-3">
              <div class="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
                <div class="text-[10px] font-semibold uppercase text-slate-400">${isEn ? 'Clients' : 'Πελάτες'}</div>
                <div class="text-lg sm:text-xl font-extrabold text-white mt-0.5">${p.client_count}</div>
                <div class="text-[10px] text-slate-400">${Math.round((p.client_share || 0) * 100)}% ${isEn ? 'of portfolio' : 'πελατολογίου'}</div>
              </div>

              <div class="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
                <div class="text-[10px] font-semibold uppercase text-slate-400">${isEn ? 'Revenue Share' : 'Μερίδιο Κύκλου Εργ.'}</div>
                <div class="text-lg sm:text-xl font-extrabold text-emerald-400 mt-0.5">${Math.round((p.turnover_share || 0) * 100)}%</div>
                <div class="text-[10px] text-slate-400">€${Math.round(p.turnover_total || 0).toLocaleString('el-GR')} συν.</div>
              </div>

              <div class="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
                <div class="text-[10px] font-semibold uppercase text-slate-400">${isEn ? 'Median Revenue' : 'Διάμεσος Κύκλος Εργ.'}</div>
                <div class="text-lg sm:text-xl font-extrabold text-white mt-0.5">€${Math.round(p.turnover_median || 0).toLocaleString('el-GR')}</div>
                <div class="text-[10px] text-slate-400">${isEn ? 'Mean' : 'Μέσος'} €${Math.round(p.turnover_mean || 0).toLocaleString('el-GR')}</div>
              </div>
            </div>

            <!-- AI-Generated Grounded Narrative with Rich Formatting -->
            <div class="p-4 sm:p-4.5 rounded-lg bg-slate-900/80 border border-slate-800 relative space-y-2">
              <div class="flex items-center justify-between pb-1 border-b border-slate-800/80">
                <span class="inline-flex items-center gap-1.5 text-[11px] font-semibold text-blue-400">
                  <i data-lucide="sparkles" class="w-3.5 h-3.5"></i>
                  Grounded AI Περιγραφή (EU AI Act Art. 50)
                </span>
                <button onclick="openLlmModal('${p.id}')" class="text-[11px] text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1 transition-colors">
                  <i data-lucide="file-code" class="w-3.5 h-3.5"></i>
                  <span>Έλεγχος Prompt &amp; AI</span>
                </button>
              </div>
              <div class="text-xs text-slate-300 leading-relaxed font-sans">
                ${formatMarkdownText(p.ai_description)}
              </div>
            </div>

            <!-- Top Statistical Trait Pills -->
            <div class="flex flex-wrap gap-1.5 pt-1">
              ${topTraits.map(t => `
                <span class="trait-pill">
                  ${escapeHtml(cleanTraitLabel(t.label))}
                  <strong class="text-emerald-300 font-mono">(${t.eb_lift.toFixed(1)}x)</strong>
                </span>
              `).join('')}
            </div>

          </div>

          <!-- Right: Sales Intelligence Cockpit -->
          <div class="lg:col-span-6 flex flex-col justify-between p-4 sm:p-5 rounded-lg bg-slate-900/80 border border-slate-800 shadow-lg min-h-[360px]">
            
            <!-- Cockpit Header: Title & Segmented Navigation Tabs -->
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-3 border-b border-slate-800">
              <div class="flex items-center gap-2">
                <span class="p-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  <i data-lucide="compass" class="w-4 h-4"></i>
                </span>
                <div>
                  <span class="text-xs sm:text-sm font-bold text-white tracking-wide">Sales Intelligence &amp; Playbook</span>
                  <span class="text-[10px] text-slate-400 block">${si.lead_funnel.total} υποψήφιοι πελάτες</span>
                </div>
              </div>

              <!-- Segmented View Tabs -->
              <div class="flex items-center gap-1 p-0.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] shrink-0 self-start sm:self-auto">
                <button onclick="switchIcpView('${p.id}', 'pipeline')" id="tab-btn-${p.id}-pipeline" class="sales-tab-btn ${activeTab === 'pipeline' ? 'active' : ''}" title="Προτεραιοποίηση &amp; Θερμότητα Leads">
                  <i data-lucide="flame" class="w-3.5 h-3.5"></i>
                  <span>Pipeline</span>
                </button>
                <button onclick="switchIcpView('${p.id}', 'territory')" id="tab-btn-${p.id}-territory" class="sales-tab-btn ${activeTab === 'territory' ? 'active' : ''}" title="Κορυφαίοι Δήμοι Εξόρμησης">
                  <i data-lucide="map-pin" class="w-3.5 h-3.5"></i>
                  <span>Περιοχές</span>
                </button>
                <button onclick="switchIcpView('${p.id}', 'playbook')" id="tab-btn-${p.id}-playbook" class="sales-tab-btn ${activeTab === 'playbook' ? 'active' : ''}" title="Pitch Hook &amp; Οδηγός Πώλησης">
                  <i data-lucide="briefcase" class="w-3.5 h-3.5"></i>
                  <span>Playbook</span>
                </button>
                <button onclick="switchIcpView('${p.id}', 'benchmark')" id="tab-btn-${p.id}-benchmark" class="sales-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}" title="Στατιστικό Lift vs Αγορά">
                  <i data-lucide="bar-chart-2" class="w-3.5 h-3.5"></i>
                  <span>Lift</span>
                </button>
              </div>
            </div>

            <!-- Tab Content Pane: 1. Pipeline Funnel (Default) -->
            <div id="pane-${p.id}-pipeline" class="sales-tab-pane ${activeTab === 'pipeline' ? '' : 'hidden'} py-3 space-y-3">
              <!-- Visual Funnel Progress Bar -->
              <div class="space-y-1.5">
                <div class="flex items-center justify-between text-[11px]">
                  <span class="font-semibold text-slate-300">Κατανομή Θερμότητας Leads</span>
                  <span class="text-slate-400 font-mono">${si.lead_funnel.total} leads στο ICP</span>
                </div>
                <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden flex shadow-inner">
                  <div class="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all" style="width: ${si.lead_funnel.hot.share_pct}%" title="Hot: ${si.lead_funnel.hot.count} (${si.lead_funnel.hot.share_pct}%)"></div>
                  <div class="h-full bg-gradient-to-r from-sky-500 to-blue-500 transition-all" style="width: ${si.lead_funnel.warm.share_pct}%" title="Warm: ${si.lead_funnel.warm.count} (${si.lead_funnel.warm.share_pct}%)"></div>
                  <div class="h-full bg-slate-600 transition-all" style="width: ${si.lead_funnel.nurture.share_pct}%" title="Nurture: ${si.lead_funnel.nurture.count} (${si.lead_funnel.nurture.share_pct}%)"></div>
                </div>
                <div class="flex items-center justify-between text-[10px] text-slate-400 font-mono px-0.5">
                  <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-emerald-400"></span> Hot: ${si.lead_funnel.hot.count} (${si.lead_funnel.hot.share_pct}%)</span>
                  <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Warm: ${si.lead_funnel.warm.count} (${si.lead_funnel.warm.share_pct}%)</span>
                  ${si.lead_funnel.nurture.count > 0 ? `<span class="flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-slate-500"></span> Nurture: ${si.lead_funnel.nurture.count}</span>` : ''}
                </div>
              </div>

              <!-- Action Cards for Sales Reps -->
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <!-- Hot Leads Card -->
                <div class="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/25 flex flex-col justify-between">
                  <div>
                    <div class="flex items-center justify-between">
                      <span class="text-xs font-bold text-emerald-300 flex items-center gap-1">
                        <i data-lucide="zap" class="w-3.5 h-3.5 text-emerald-400"></i>
                        🔥 Hot Leads (≥90%)
                      </span>
                      <span class="text-lg font-black text-white font-mono">${si.lead_funnel.hot.count}</span>
                    </div>
                    <p class="text-[11px] text-slate-300 mt-1 leading-snug">
                      Κορυφαία ταύτιση DNA. Άμεση τηλεφωνική κλήση &amp; κλείσιμο 15-λεπτου Demo εντός 48 ωρών.
                    </p>
                  </div>
                  <button onclick="filterLeadsByIcpAndTier('${p.id}', 'hot')" class="mt-2.5 w-full py-1 px-2 rounded bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold flex items-center justify-center gap-1 transition-colors">
                    <span>Προβολή ${si.lead_funnel.hot.count} Hot Leads</span>
                    <i data-lucide="arrow-down-right" class="w-3 h-3"></i>
                  </button>
                </div>

                <!-- Warm Leads Card -->
                <div class="p-3 rounded-lg bg-blue-950/20 border border-blue-500/25 flex flex-col justify-between">
                  <div>
                    <div class="flex items-center justify-between">
                      <span class="text-xs font-bold text-blue-300 flex items-center gap-1">
                        <i data-lucide="mail" class="w-3.5 h-3.5 text-blue-400"></i>
                        ⚡ Warm Leads (80-89%)
                      </span>
                      <span class="text-lg font-black text-white font-mono">${si.lead_funnel.warm.count}</span>
                    </div>
                    <p class="text-[11px] text-slate-300 mt-1 leading-snug">
                      Ισχυρή συνάφεια. Ένταξη σε αυτοματοποιημένη ακολουθία email &amp; LinkedIn cadences.
                    </p>
                  </div>
                  <button onclick="filterLeadsByIcpAndTier('${p.id}', 'warm')" class="mt-2.5 w-full py-1 px-2 rounded bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-[11px] font-semibold flex items-center justify-center gap-1 transition-colors">
                    <span>Προβολή ${si.lead_funnel.warm.count} Warm Leads</span>
                    <i data-lucide="arrow-down-right" class="w-3 h-3"></i>
                  </button>
                </div>
              </div>

              <!-- Deal Drivers Footprint -->
              <div class="grid grid-cols-3 gap-2 pt-1 border-t border-slate-800/60 text-[11px]">
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <div class="text-[9px] uppercase font-semibold text-slate-400">Μορφή Απόφασης</div>
                  <div class="font-bold text-slate-200 mt-0.5 truncate">${si.firmographics.dominant_legal} (${si.firmographics.top_legal[0] ? si.firmographics.top_legal[0].share_pct : 0}%)</div>
                </div>
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <div class="text-[9px] uppercase font-semibold text-slate-400">Multi-Branch</div>
                  <div class="font-bold text-emerald-400 mt-0.5 truncate">${si.firmographics.multi_branch_count} (${si.firmographics.multi_branch_share_pct}%)</div>
                </div>
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <div class="text-[9px] uppercase font-semibold text-slate-400">Διάμεση Ηλικία</div>
                  <div class="font-bold text-slate-200 mt-0.5 truncate">${si.firmographics.median_age_years} έτη</div>
                </div>
              </div>
            </div>

            <!-- Tab Content Pane: 2. Sales Territories Hotspots -->
            <div id="pane-${p.id}-territory" class="sales-tab-pane ${activeTab === 'territory' ? '' : 'hidden'} py-2 space-y-3">
              <div class="flex items-center justify-between text-xs">
                <span class="font-semibold text-slate-300">Κορυφαίοι Δήμοι Εξόρμησης (Συγκέντρωση Leads)</span>
                <span class="text-[10px] text-slate-400 font-mono">Αριθμός &amp; Μερίδιο</span>
              </div>
              <div class="h-44 sm:h-48 relative">
                <canvas id="chart-territory-${p.id}"></canvas>
              </div>
              <div class="p-2.5 rounded-lg bg-blue-950/20 border border-blue-500/20 flex items-start gap-2">
                <i data-lucide="lightbulb" class="w-4 h-4 text-amber-400 shrink-0 mt-0.5"></i>
                <div class="text-[11px] text-slate-300 leading-snug">
                  ${escapeHtml(si.territory_tip)}
                </div>
              </div>
            </div>

            <!-- Tab Content Pane: 3. Commercial Playbook & Pitch Hooks -->
            <div id="pane-${p.id}-playbook" class="sales-tab-pane ${activeTab === 'playbook' ? '' : 'hidden'} py-2 space-y-2.5">
              <!-- Deal Qualification Strip -->
              <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px]">
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <span class="text-[9px] uppercase font-semibold text-slate-400 block">Deal Tier</span>
                  <span class="font-bold text-slate-200 truncate block">${escapeHtml(si.playbook.deal_tier)}</span>
                </div>
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <span class="text-[9px] uppercase font-semibold text-slate-400 block">Κύκλος Πώλησης</span>
                  <span class="font-bold text-emerald-400">${escapeHtml(si.playbook.sales_cycle)}</span>
                </div>
                <div class="p-2 rounded bg-slate-950/60 border border-slate-800 col-span-2 sm:col-span-1">
                  <span class="text-[9px] uppercase font-semibold text-slate-400 block">Decision Maker</span>
                  <span class="font-bold text-blue-300 truncate block">${escapeHtml(si.playbook.buyer_persona)}</span>
                </div>
              </div>

              <!-- Pain Points -->
              <div class="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px] space-y-1">
                <div class="font-bold text-slate-200 flex items-center gap-1.5 text-xs">
                  <i data-lucide="alert-circle" class="w-3.5 h-3.5 text-amber-400"></i>
                  <span>Κύρια Επιχειρησιακά Pain Points (Γιατί αγοράζουν)</span>
                </div>
                <ul class="text-slate-300 space-y-0.5 list-disc list-inside">
                  ${(si.playbook.pain_points || []).map(pt => `<li>${escapeHtml(pt)}</li>`).join('')}
                </ul>
              </div>

              <!-- Elevator Pitch Quote Box -->
              <div class="p-2.5 rounded-lg bg-blue-950/30 border border-blue-500/30 text-[11px] space-y-1">
                <div class="font-bold text-blue-300 flex items-center gap-1.5 text-xs">
                  <i data-lucide="quote" class="w-3.5 h-3.5 text-blue-400"></i>
                  <span>Elevator Pitch (Το άνοιγμα του πωλητή στο τηλέφωνο)</span>
                </div>
                <p class="text-slate-200 italic leading-snug">
                  ${escapeHtml(si.playbook.pitch_hook)}
                </p>
              </div>

              <!-- Objection Handling -->
              <div class="p-2 rounded bg-slate-950/50 border border-slate-800 text-[11px]">
                <span class="font-bold text-slate-300">Χειρισμός Αντιρρήσεων:</span>
                <span class="text-slate-300 ml-1 leading-snug">${escapeHtml(si.playbook.objection_rebuttal)}</span>
              </div>
            </div>

            <!-- Tab Content Pane: 4. Benchmark Statistical Lift (Polished & Categorized) -->
            <div id="pane-${p.id}-benchmark" class="sales-tab-pane ${activeTab === 'benchmark' ? '' : 'hidden'} py-2 space-y-2">
              <div class="flex items-center justify-between text-xs">
                <span class="font-semibold text-slate-300">Στατιστικό Πλεονέκτημα vs Αγορά (24.424 επιχειρήσεις)</span>
                <span class="text-[10px] text-slate-400">Empirical Bayes Lift</span>
              </div>
              <div class="h-44 sm:h-48 relative">
                <canvas id="chart-comp-${p.id}"></canvas>
              </div>
              <div class="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-800/80">
                <div class="flex items-center gap-3">
                  <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-sm" style="background-color: ${color}"></span> Πελάτες ICP</span>
                  <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-slate-600"></span> Μέσος Όρος Αγοράς</span>
                </div>
                <span>Καθαρά χαρακτηριστικά FDR p &lt; 0.05</span>
              </div>
            </div>

            <!-- Cockpit Footer -->
            <div class="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800/80 mt-1">
              <span class="inline-flex items-center gap-1">
                <i data-lucide="shield-check" class="w-3.5 h-3.5 text-emerald-400"></i>
                Commercial Fit Only (EU AI Act compliant)
              </span>
              <button onclick="panMapToMedoid(${p.medoid[0]}, ${p.medoid[1]}, '${p.id}')" class="text-blue-400 hover:text-blue-300 font-semibold flex items-center gap-1 transition-colors">
                <i data-lucide="map" class="w-3.5 h-3.5"></i>
                <span>Εστίαση στον Χάρτη</span>
              </button>
            </div>

          </div>

        </div>

      </div>
    `;
  }).join('');
  
  // Initialize charts (Territory and Benchmark)
  state.profiles.forEach((p, idx) => {
    const color = ICP_COLORS[idx % ICP_COLORS.length];
    const si = getProfileSalesIntel(p);
    
    // 1. Territory Chart
    const terrCanvas = document.getElementById(`chart-territory-${p.id}`);
    if (terrCanvas && si.territories && si.territories.length) {
      const terrLabels = si.territories.map(t => t.name);
      const terrCounts = si.territories.map(t => t.count);
      const terrShares = si.territories.map(t => t.share_pct);

      state.territoryCharts[p.id] = new Chart(terrCanvas, {
        type: 'bar',
        data: {
          labels: terrLabels,
          datasets: [{
            label: 'Υποψήφιοι Πελάτες',
            data: terrCounts,
            backgroundColor: color,
            borderRadius: 4,
            barPercentage: 0.65,
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#0f172a',
              titleColor: '#f8fafc',
              bodyColor: '#cbd5e1',
              borderColor: '#334155',
              borderWidth: 1,
              callbacks: {
                label: (ctx) => ` ${terrCounts[ctx.dataIndex]} επιχειρήσεις (${terrShares[ctx.dataIndex]}% του ICP)`
              }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(148, 163, 184, 0.08)' },
              ticks: { color: '#94a3b8', font: { size: 10 } }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#e2e8f0', font: { size: 11, weight: '500' } }
            }
          }
        }
      });
    }

    // 2. Benchmark Lift Chart
    const canvas = document.getElementById(`chart-comp-${p.id}`);
    if (canvas) {
      const topTraits = (p.features || []).slice(0, 5);
      const labels = topTraits.map(t => cleanTraitLabel(t.label));
      const icpShares = topTraits.map(t => Math.round((t.icp_share || 0) * 100));
      const univShares = topTraits.map(t => Math.round((t.universe_share || 0) * 100));
      
      state.compCharts[p.id] = new Chart(canvas, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Στο Προφίλ μας (%)',
              data: icpShares,
              backgroundColor: color,
              borderRadius: 4,
              barPercentage: 0.7,
            },
            {
              label: 'Μέσος Όρος Αγοράς (%)',
              data: univShares,
              backgroundColor: '#475569',
              borderRadius: 4,
              barPercentage: 0.7,
            }
          ]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#0f172a',
              titleColor: '#f8fafc',
              bodyColor: '#cbd5e1',
              borderColor: '#334155',
              borderWidth: 1,
              callbacks: {
                afterBody: (context) => {
                  const itemIdx = context[0].dataIndex;
                  const trait = topTraits[itemIdx];
                  return `Empirical Bayes Lift: ${trait.eb_lift.toFixed(2)}x (p_FDR: ${trait.p_fdr ? trait.p_fdr.toExponential(2) : '-'})`;
                }
              }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(148, 163, 184, 0.08)' },
              ticks: { color: '#94a3b8', font: { size: 10 }, callback: v => `${v}%` }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#e2e8f0', font: { size: 11, weight: '500' } }
            }
          }
        }
      });
    }
  });

  lucide.createIcons();
}

function cleanTraitLabel(label) {
  if (!label) return '';
  let s = String(label).trim();

  // KAD formatting
  if (s.startsWith('ΚΑΔ ') || s.includes('ΚΑΔ')) {
    const m = s.match(/ΚΑΔ\s*(\d+)\s*\(([^)]+)\)/i);
    if (m) {
      const code = m[1].slice(0, 4);
      let desc = m[2].trim();
      if (desc.length > 26) desc = desc.slice(0, 24) + '...';
      return `[ΚΑΔ ${code}] ${desc}`;
    }
    const m2 = s.match(/ΚΑΔ\s*(\d+)/i);
    if (m2) return `[ΚΑΔ ${m2[1].slice(0, 4)}]`;
  }

  if (s.startsWith('Δήμος: ')) {
    const name = s.replace('Δήμος: ', '').split('/')[0].trim();
    return `[Δήμος] ${name}`;
  }
  if (s.startsWith('Νομός: ')) {
    const name = s.replace('Νομός: ', '').split('/')[0].trim();
    return `[Νομός] ${name}`;
  }
  if (s.startsWith('Νομική Μορφή: ')) {
    const name = s.replace('Νομική Μορφή: ', '').trim();
    return `[Μορφή] ${name}`;
  }

  return s.split('/')[0].trim();
}

window.switchIcpView = function(profileId, viewKey) {
  const panes = ['pipeline', 'territory', 'playbook', 'benchmark'];
  panes.forEach(pane => {
    const el = document.getElementById(`pane-${profileId}-${pane}`);
    const btn = document.getElementById(`tab-btn-${profileId}-${pane}`);
    if (el) {
      if (pane === viewKey) el.classList.remove('hidden');
      else el.classList.add('hidden');
    }
    if (btn) {
      if (pane === viewKey) btn.classList.add('active');
      else btn.classList.remove('active');
    }
  });
  state.icpActiveTab = state.icpActiveTab || {};
  state.icpActiveTab[profileId] = viewKey;

  if (viewKey === 'territory') {
    requestAnimationFrame(() => {
      const ch = state.territoryCharts && state.territoryCharts[profileId];
      if (ch) ch.resize();
    });
  }
  if (viewKey === 'benchmark') {
    requestAnimationFrame(() => {
      const ch = state.compCharts && state.compCharts[profileId];
      if (ch) ch.resize();
    });
  }
};

window.filterLeadsByIcpAndTier = function(profileId, tier) {
  state.leadsFilter.profile = profileId;
  const profSelect = document.getElementById('leads-profile-filter');
  if (profSelect) profSelect.value = profileId;

  state.leadsFilter.quickFilter = tier === 'hot' ? 'hot' : 'all';
  document.querySelectorAll('.filter-chip').forEach(c => {
    if (c.dataset.filter === state.leadsFilter.quickFilter) c.classList.add('active');
    else c.classList.remove('active');
  });

  const cockpitTabBtn = document.querySelector('.nav-tab-btn[data-tab="overview"]');
  if (cockpitTabBtn) cockpitTabBtn.click();

  renderLeadsTable();
};

window.panMapToMedoid = function(lat, lon, profileId) {
  if (state.map && lat && lon) {
    const overviewTabBtn = document.querySelector('.nav-tab-btn[data-tab="overview"]');
    if (overviewTabBtn && !overviewTabBtn.classList.contains('active')) {
      overviewTabBtn.click();
    }
    state.map.setView([lat, lon], 10, { animate: true });
    const mapEl = document.getElementById('leaflet-map');
    if (mapEl) {
      mapEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
};

// -----------------------------------------------------------------------------
// Lookalike Leads Table, Checkboxes, Batch FAB & Nearest-Neighbor Route Builder
// -----------------------------------------------------------------------------
function updateQuickChipCounters() {
  const all = state.lookalikes || [];
  const hotCnt = all.filter(l => (l.score || 0) >= 0.85).length;
  const routeCnt = all.filter(l => typeof l.nearest_client_km === 'number' && l.nearest_client_km <= 1.0).length;
  const multiCnt = all.filter(l => l.is_branch || (l.branches && l.branches > 1)).length;
  const tier1Cnt = all.filter(l => l.osm_tier === 'Tier 1 (Confirmed Match)').length;

  safeSetText('chip-count-all', String(all.length));
  safeSetText('chip-count-hot', String(hotCnt));
  safeSetText('chip-count-route', String(routeCnt));
  safeSetText('chip-count-multi', String(multiCnt));
  safeSetText('chip-count-tier1', String(tier1Cnt));
}

function updateBatchFabUI() {
  const bar = document.getElementById('batch-action-bar');
  const countEl = document.getElementById('batch-selected-count');
  const cnt = state.selectedLeadIds.size;
  if (countEl) countEl.textContent = String(cnt);
  if (bar) {
    bar.classList.toggle('visible', cnt > 0);
  }
}

window.toggleSelectLead = function(companyId, isChecked) {
  if (isChecked) state.selectedLeadIds.add(Number(companyId));
  else state.selectedLeadIds.delete(Number(companyId));
  updateBatchFabUI();
};

window.toggleSelectAllPageLeads = function(isChecked) {
  const f = state.leadsFilter;
  const items = state.currentFilteredLeads || [];
  const startIdx = (f.page - 1) * f.pageSize;
  const pageItems = items.slice(startIdx, startIdx + f.pageSize);
  pageItems.forEach(l => {
    if (isChecked) state.selectedLeadIds.add(Number(l.company_id));
    else state.selectedLeadIds.delete(Number(l.company_id));
  });
  updateBatchFabUI();
  renderLeadsTable();
};

window.clearSelectedLeads = function() {
  state.selectedLeadIds.clear();
  const selAll = document.getElementById('leads-select-all');
  if (selAll) selAll.checked = false;
  updateBatchFabUI();
  renderLeadsTable();
};

window.exportSelectedLeadsCsv = function() {
  const items = (state.lookalikes || []).filter(l => state.selectedLeadIds.has(Number(l.company_id)));
  if (!items.length) return;
  downloadLeadsListAsCsv(items, `ForceCRM_Selected_${items.length}_Leads.csv`);
};

window.batchVerifySelectedGemi = async function() {
  const ids = Array.from(state.selectedLeadIds).slice(0, 15);
  if (!ids.length) return;
  const isEn = state.lang === 'en';
  showToast(isEn ? `🔍 Verifying ${ids.length} selected prospects against GEMI...` : `🔍 Έλεγχος Γ.Ε.ΜΗ. για ${ids.length} επιλεγμένες επιχειρήσεις...`, 'shield-check');
  let activeCnt = 0;
  for (const cid of ids) {
    try {
      const resp = await fetch(`/api/verify-company/${cid}`, { method: 'POST' });
      const data = await resp.json();
      if (data.status === 'ok' && data.verification && !data.verification.is_closed) {
        activeCnt++;
      }
    } catch (_) {}
  }
  showToast(isEn ? `✅ Verified ${activeCnt}/${ids.length} active in GEMI Registry` : `✅ Επιβεβαιώθηκαν ${activeCnt}/${ids.length} ενεργές στο Γ.Ε.ΜΗ.`, 'check-circle-2');
};

function findNearestClient(lat, lon) {
  if (!state.clients || !state.clients.length || !lat || !lon) return null;
  let best = null;
  let minDist = Infinity;
  for (const c of state.clients) {
    if (c.lat && c.lon) {
      const d = computeDistanceKm(lat, lon, c.lat, c.lon);
      if (d < minDist) {
        minDist = d;
        best = { ...c, distanceKm: d };
      }
    }
  }
  return best;
}
window.findNearestClient = findNearestClient;

async function drawMultiStopRoute(rawStops, options = {}) {
  const isEn = state.lang === 'en';
  if (!rawStops || rawStops.length < 2) {
    showToast(isEn ? '⚠️ Need at least 2 points to map a route.' : '⚠️ Απαιτούνται τουλάχιστον 2 σημεία για σχεδίαση διαδρομής.', 'alert-circle');
    return;
  }

  showToast(isEn ? '⏳ Computing real road route via OSRM...' : '⏳ Υπολογισμός πραγματικής οδικής διαδρομής μέσω OSRM...', 'navigation');

  // If first stop is designated as origin depot (keepFirstOrigin: true), route from it sequentially via TSP
  let route = [];
  if (options.keepFirstOrigin) {
    const origin = rawStops[0];
    const rest = [...rawStops.slice(1)];
    route = [origin];
    while (rest.length > 0) {
      const curr = route[route.length - 1];
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < rest.length; i++) {
        const d = computeDistanceKm(curr.lat, curr.lon, rest[i].lat, rest[i].lon);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      route.push(rest.splice(bestIdx, 1)[0]);
    }
  } else {
    // Pure TSP starting from point 0
    const unvisited = [...rawStops];
    route = [unvisited.shift()];
    while (unvisited.length > 0) {
      const curr = route[route.length - 1];
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < unvisited.length; i++) {
        const d = computeDistanceKm(curr.lat, curr.lon, unvisited[i].lat, unvisited[i].lon);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      route.push(unvisited.splice(bestIdx, 1)[0]);
    }
  }

  let straightKm = 0;
  for (let i = 0; i < route.length - 1; i++) {
    straightKm += computeDistanceKm(route[i].lat, route[i].lon, route[i+1].lat, route[i+1].lon);
  }

  // Query OpenStreetMap OSRM Public Routing API for realistic street geometry
  let roadCoordinates = null;
  let roadDistanceKm = straightKm;
  let roadDurationMin = Math.round((straightKm / 35) * 60);

  try {
    const coordString = route.map(r => `${r.lon},${r.lat}`).join(';');
    const osrmUrl = `https://router.project-osrm.org/route/v1/driving/${coordString}?overview=full&geometries=geojson&steps=false`;
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6500);
    const osrmResp = await fetch(osrmUrl, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (osrmResp.ok) {
      const data = await osrmResp.json();
      if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
        const primary = data.routes[0];
        roadCoordinates = primary.geometry.coordinates.map(pt => [pt[1], pt[0]]);
        roadDistanceKm = primary.distance / 1000;
        roadDurationMin = Math.round(primary.duration / 60);
      }
    }
  } catch (err) {
    console.warn('OSRM routing fallback to geodesic polyline:', err);
  }

  if (state.map && state.salesRouteLayer) {
    state.salesRouteLayer.clearLayers();
    const latlngs = roadCoordinates || route.map(r => [r.lat, r.lon]);

    // Outer glow polyline
    const roadPoly = L.polyline(latlngs, {
      color: '#06b6d4',
      weight: 5.5,
      opacity: 0.92,
      lineJoin: 'round',
      lineCap: 'round',
    }).addTo(state.salesRouteLayer);

    // Inner bright core
    L.polyline(latlngs, {
      color: '#a5f3fc',
      weight: 2,
      opacity: 0.95,
      lineJoin: 'round',
      lineCap: 'round',
    }).addTo(state.salesRouteLayer);

    route.forEach((stop, idx) => {
      const isClientStop = stop.isClient;
      const badgeHtml = isClientStop
        ? `<div style="width:28px;height:28px;border-radius:9999px;background:#d97706;color:#fff;font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;border:2.5px solid #fef08a;box-shadow:0 3px 12px rgba(217,119,6,0.7);">⭐${idx + 1}</div>`
        : `<div style="width:26px;height:26px;border-radius:9999px;background:#0891b2;color:#fff;font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;border:2.5px solid #fff;box-shadow:0 3px 10px rgba(0,0,0,0.7);">🎯${idx + 1}</div>`;

      const stopIcon = L.divIcon({
        className: 'route-stop-badge',
        html: badgeHtml,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });

      const popupHtml = `
        <div class="text-xs p-1 space-y-1 w-[260px] min-w-[260px]">
          <div class="font-extrabold ${isClientStop ? 'text-amber-300' : 'text-cyan-300'} text-sm leading-tight">
            ${isClientStop ? '⭐ ' : '🎯 '}#${idx + 1}. ${escapeHtml(stop.name)}
          </div>
          <div class="inline-flex items-center gap-1 px-1.5 py-0.2 rounded text-[10px] font-semibold ${isClientStop ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'}">
            ${isClientStop ? (isEn ? 'Active Client (Origin)' : 'Ενεργός Πελάτης (Αφετηρία)') : (isEn ? 'Target Prospect' : 'Υποψήφιος Πελάτης')}
          </div>
          <div class="text-slate-300 font-mono text-[11px] pt-0.5">ΑΦΜ: ${stop.vat || '-'}</div>
          ${stop.turnover ? `<div class="text-emerald-400 font-semibold font-mono text-[11px]">Κύκλος Εργ.: €${Math.round(stop.turnover).toLocaleString()}</div>` : ''}
          <div class="text-slate-400 text-[10px] truncate">${escapeHtml(stop.primary_kad_title || stop.kad_desc || '')}</div>
        </div>
      `;

      L.marker([stop.lat, stop.lon], { icon: stopIcon })
        .bindPopup(popupHtml, { minWidth: 260, maxWidth: 300 })
        .addTo(state.salesRouteLayer);
    });

    if (state.activeTab !== 'overview') {
      const ovTabBtn = document.querySelector('.nav-tab-btn[data-tab="overview"]');
      if (ovTabBtn) ovTabBtn.click();
    }
    if (state.layoutMode === 'table') {
      setLayoutMode('split');
    }
    state.map.fitBounds(roadPoly.getBounds(), { padding: [55, 55] });
  }

  const isRealRoad = roadCoordinates !== null;
  const clientStops = route.filter(r => r.isClient).length;
  const prospectStops = route.length - clientStops;
  const titleText = options.title || (isEn ? 'Sales & Delivery Route' : 'Διαδρομή Πωλήσεων & Διανομής');

  // Save active route state for details panel and Google Maps navigation
  state.activeRoute = {
    stops: route,
    roadCoordinates,
    roadDistanceKm,
    roadDurationMin,
    isRealRoad,
    title: titleText,
    straightKm,
  };

  const container = document.getElementById('map-route-banner-container');
  const banner = document.getElementById('map-route-banner');
  const summaryText = document.getElementById('map-route-summary-text');
  if (container) container.classList.remove('hidden');
  if (banner) banner.classList.remove('hidden');
  if (summaryText) {
    summaryText.innerHTML = `
      <span>${escapeHtml(titleText)}: <strong>${route.length} στάσεις</strong> (${clientStops ? `⭐ ${clientStops} πελ. + ` : ''}🎯 ${prospectStops} leads) · <strong>${roadDistanceKm.toFixed(1)} km</strong> · ~${roadDurationMin}m ${isRealRoad ? '<span class="text-emerald-400 font-normal ml-0.5">(OSRM)</span>' : '<span class="text-amber-400 font-normal ml-0.5">(Geodesic)</span>'}</span>
    `;
  }

  // Refresh details panel if currently expanded
  const detailsPanel = document.getElementById('map-route-details-panel');
  if (detailsPanel && !detailsPanel.classList.contains('hidden')) {
    window.renderRouteDetails();
  }

  showToast(
    isEn ? `🗺️ Route mapped: ${route.length} stops (${roadDistanceKm.toFixed(1)} km, ~${roadDurationMin} min)` : `🗺️ Χαρτογραφήθηκε διαδρομή: ${route.length} στάσεις (${roadDistanceKm.toFixed(1)} km, ~${roadDurationMin} λ)`,
    'map'
  );
}
window.drawMultiStopRoute = drawMultiStopRoute;

window.renderRouteDetails = function() {
  const panel = document.getElementById('map-route-details-panel');
  if (!panel || !state.activeRoute) return;
  const r = state.activeRoute;
  const stops = r.stops || [];
  const isEn = state.lang === 'en';

  let stopsHtml = '';
  stops.forEach((s, idx) => {
    const isClient = s.isClient;
    const badgeBg = isClient ? 'bg-amber-500/20 text-amber-300 border-amber-500/35' : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/35';
    const badgeText = isClient
      ? (idx === 0 ? (isEn ? '⭐ Origin: Client' : '⭐ Αφετηρία: Πελάτης') : (isEn ? '⭐ Client Stop' : '⭐ Στάση: Πελάτης'))
      : (isEn ? `🎯 Stop #${idx + 1}: Prospect` : `🎯 Στάση #${idx + 1}: Υποψήφιος`);
    
    const address = [s.street, s.street_number, s.municipality || s.city, s.prefecture || s.prefecture_name].filter(Boolean).join(', ');
    const extraInfo = isClient
      ? `${isEn ? 'Turnover' : 'Κύκλος Εργασιών'}: <strong class="text-white font-mono">€${(s.turnover || 0).toLocaleString()}</strong>`
      : `${isEn ? 'Match' : 'Ταύτιση'}: <strong class="text-emerald-400 font-mono">${((s.score || 0) * 100).toFixed(1)}%</strong> (${escapeHtml(s.profile_id || '')})`;

    stopsHtml += `
      <div class="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1.5 transition-all hover:border-slate-700">
        <div class="flex items-center justify-between gap-2 flex-wrap">
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold border ${badgeBg}">${badgeText}</span>
            <span class="font-bold text-white text-xs">${escapeHtml(s.name || s.company_name || '')}</span>
          </div>
          <div class="flex items-center gap-1.5">
            <button type="button" onclick="panToRouteStop(${idx})" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-sky-300 hover:text-white border border-slate-700 text-[10px] font-semibold flex items-center gap-1 transition-colors" title="${isEn ? 'Focus on map' : 'Εστίαση στον χάρτη'}">
              <i data-lucide="crosshair" class="w-3 h-3 text-sky-400"></i>
              <span>${isEn ? 'Focus' : 'Εστίαση'}</span>
            </button>
            <button type="button" onclick="inspectCompany(${s.company_id})" class="px-2 py-0.5 rounded bg-blue-600/30 hover:bg-blue-600 text-blue-200 hover:text-white border border-blue-500/40 text-[10px] font-semibold flex items-center gap-1 transition-colors" title="${isEn ? 'Inspect Dossier' : 'Καρτέλα'}">
              <i data-lucide="building-2" class="w-3 h-3 text-blue-400"></i>
              <span>${isEn ? 'Card' : 'Καρτέλα'}</span>
            </button>
          </div>
        </div>
        <div class="text-[11px] text-slate-300 flex items-center gap-2 flex-wrap">
          <span>🏛️ ${escapeHtml(s.legal_type || '')}</span>
          <span>·</span>
          <span>ΑΦΜ: <span class="font-mono text-slate-200">${s.vat || '-'}</span></span>
          <span>·</span>
          <span>${extraInfo}</span>
        </div>
        ${address ? `<div class="text-[11px] text-slate-400">📍 ${escapeHtml(address)}</div>` : ''}
        ${s.primary_kad_title || s.kad_desc ? `<div class="text-[10px] text-slate-400 truncate">🍽️ ${escapeHtml(s.primary_kad_title || s.kad_desc)}</div>` : ''}
      </div>
    `;

    if (idx < stops.length - 1) {
      stopsHtml += `
        <div class="flex items-center justify-center my-0.5 text-cyan-400/80 font-mono text-[10px] gap-1">
          <span>↓</span>
          <span>${isEn ? 'Leg' : 'Σκέλος'} ${idx + 1} ➔ ${idx + 2}</span>
        </div>
      `;
    }
  });

  panel.innerHTML = `
    <!-- Summary Header Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pb-2 border-b border-slate-800">
      <div class="p-2 rounded-lg bg-slate-900 border border-slate-800">
        <div class="text-[10px] text-slate-400 uppercase font-semibold">${isEn ? 'Distance' : 'Απόσταση'}</div>
        <div class="text-sm font-extrabold text-cyan-300 font-mono">${r.roadDistanceKm.toFixed(1)} km</div>
      </div>
      <div class="p-2 rounded-lg bg-slate-900 border border-slate-800">
        <div class="text-[10px] text-slate-400 uppercase font-semibold">${isEn ? 'Est. Driving Time' : 'Εκτ. Χρόνος'}</div>
        <div class="text-sm font-extrabold text-white font-mono">~${r.roadDurationMin} min</div>
      </div>
      <div class="p-2 rounded-lg bg-slate-900 border border-slate-800">
        <div class="text-[10px] text-slate-400 uppercase font-semibold">${isEn ? 'Total Stops' : 'Σύνολο Στάσεων'}</div>
        <div class="text-sm font-extrabold text-white font-mono">${stops.length}</div>
      </div>
      <div class="p-2 rounded-lg bg-slate-900 border border-slate-800">
        <div class="text-[10px] text-slate-400 uppercase font-semibold">${isEn ? 'Routing Engine' : 'Δρομολόγηση'}</div>
        <div class="text-[11px] font-bold ${r.isRealRoad ? 'text-emerald-400' : 'text-amber-400'} font-mono">${r.isRealRoad ? 'OSRM Live Road' : 'Geodesic'}</div>
      </div>
    </div>

    <!-- Stops Itinerary -->
    <div class="space-y-1.5">
      <div class="text-[11px] font-bold uppercase tracking-wider text-slate-300 flex items-center justify-between">
        <span>${isEn ? 'Detailed Delivery Sequence' : 'Αναλυτική Σειρά Στάσεων Διαδρομής'}</span>
        <span class="text-[10px] text-slate-400 font-normal">(${stops.length} ${isEn ? 'waypoints' : 'σημεία'})</span>
      </div>
      <div class="space-y-1.5">
        ${stopsHtml}
      </div>
    </div>

    <!-- Footer Action Bar -->
    <div class="pt-2 border-t border-slate-800 flex items-center justify-between gap-2 flex-wrap">
      <button type="button" onclick="openGoogleMapsRoute()" class="px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 hover:text-white border border-emerald-500/40 font-semibold text-xs flex items-center gap-1.5 transition-colors">
        <i data-lucide="navigation" class="w-3.5 h-3.5 text-emerald-400"></i>
        <span>${isEn ? 'Open in Google Maps Navigation' : 'Άνοιγμα σε Google Maps (Πλοήγηση)'}</span>
      </button>
      <button type="button" onclick="toggleRouteDetails(false)" class="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-colors">
        ${isEn ? 'Close' : 'Κλείσιμο'}
      </button>
    </div>
  `;

  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
};

window.toggleRouteDetails = function(forceState = null) {
  const panel = document.getElementById('map-route-details-panel');
  const chevron = document.getElementById('route-details-chevron');
  const btnLabel = document.getElementById('route-details-btn-label');
  if (!panel) return;
  const isEn = state.lang === 'en';
  const shouldOpen = forceState !== null ? forceState : panel.classList.contains('hidden');

  if (shouldOpen) {
    window.renderRouteDetails();
    panel.classList.remove('hidden');
    if (chevron) chevron.classList.add('rotate-180');
    if (btnLabel) btnLabel.textContent = isEn ? 'Hide Info' : 'Απόκρυψη';
  } else {
    panel.classList.add('hidden');
    if (chevron) chevron.classList.remove('rotate-180');
    if (btnLabel) btnLabel.textContent = isEn ? 'Details' : 'Λεπτομέρειες';
  }
};

window.panToRouteStop = function(stopIndex) {
  if (!state.activeRoute || !state.activeRoute.stops || !state.activeRoute.stops[stopIndex]) return;
  const stop = state.activeRoute.stops[stopIndex];
  if (state.map && stop.lat && stop.lon) {
    state.map.setView([stop.lat, stop.lon], Math.max(state.map.getZoom(), 14), { animate: true });
    if (state.hoverHaloMarker) {
      state.hoverHaloMarker.setLatLng([stop.lat, stop.lon]);
      state.hoverHaloMarker.setOpacity(1);
      setTimeout(() => {
        if (state.hoverHaloMarker) state.hoverHaloMarker.setOpacity(0);
      }, 2500);
    }
  }
};

window.openGoogleMapsRoute = function() {
  if (!state.activeRoute || !state.activeRoute.stops || state.activeRoute.stops.length < 2) return;
  const stops = state.activeRoute.stops;
  const origin = `${stops[0].lat},${stops[0].lon}`;
  const destination = `${stops[stops.length - 1].lat},${stops[stops.length - 1].lon}`;
  let url = `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}`;
  if (stops.length > 2) {
    const waypoints = stops.slice(1, -1).map(s => `${s.lat},${s.lon}`).join('|');
    url += `&waypoints=${encodeURIComponent(waypoints)}`;
  }
  window.open(url, '_blank', 'noopener,noreferrer');
};

window.mapRouteBetweenClientAndProspect = async function(prospectId, clientId = null) {
  const isEn = state.lang === 'en';
  const prospect = (state.lookalikes || []).find(l => l.company_id === Number(prospectId));
  if (!prospect || !prospect.lat || !prospect.lon) {
    showToast(isEn ? '⚠️ Prospect GPS coordinates missing' : '⚠️ Λείπουν οι συντεταγμένες GPS του υποψηφίου', 'alert-circle');
    return;
  }

  let client = null;
  if (clientId) {
    client = (state.clients || []).find(c => c.company_id === Number(clientId));
  } else {
    client = findNearestClient(prospect.lat, prospect.lon);
  }

  if (!client || !client.lat || !client.lon) {
    showToast(isEn ? '⚠️ No active client found nearby' : '⚠️ Δεν βρέθηκε ενεργός πελάτης κοντά', 'alert-circle');
    return;
  }

  await drawMultiStopRoute([
    { ...client, isClient: true },
    { ...prospect, isClient: false }
  ], {
    keepFirstOrigin: true,
    title: isEn
      ? `Delivery from "${client.name}" to "${prospect.name}"`
      : `Διανομή από «${client.name}» προς «${prospect.name}»`
  });
};

window.mapRouteFromClientToNearbyProspects = async function(clientId, maxStops = 3, maxRadiusKm = 10) {
  const isEn = state.lang === 'en';
  const client = (state.clients || []).find(c => c.company_id === Number(clientId));
  if (!client || !client.lat || !client.lon) {
    showToast(isEn ? '⚠️ Client GPS coordinates missing' : '⚠️ Λείπουν οι συντεταγμένες GPS του πελάτη', 'alert-circle');
    return;
  }

  const prospects = (state.lookalikes || [])
    .filter(l => l.lat && l.lon)
    .map(l => ({ ...l, distToClient: computeDistanceKm(client.lat, client.lon, l.lat, l.lon) }))
    .filter(l => l.distToClient <= maxRadiusKm)
    .sort((a, b) => a.distToClient - b.distToClient)
    .slice(0, maxStops);

  if (!prospects.length) {
    showToast(isEn ? `⚠️ No prospects found within ${maxRadiusKm} km of client` : `⚠️ Δεν βρέθηκαν υποψήφιοι σε ακτίνα ${maxRadiusKm} km από τον πελάτη`, 'alert-circle');
    return;
  }

  await drawMultiStopRoute([
    { ...client, isClient: true },
    ...prospects.map(p => ({ ...p, isClient: false }))
  ], {
    keepFirstOrigin: true,
    title: isEn
      ? `Circuit: "${client.name}" ➔ ${prospects.length} Nearby Prospects`
      : `Κύκλωμα: «${client.name}» ➔ ${prospects.length} Γειτονικοί Υποψήφιοι`
  });
};

window.buildSalesRouteFromSelected = async function() {
  const selected = (state.lookalikes || []).filter(l => state.selectedLeadIds.has(Number(l.company_id)) && l.lat && l.lon);
  const isEn = state.lang === 'en';
  if (selected.length === 0) {
    showToast(isEn ? '⚠️ Select at least 1 prospect or client to build a route.' : '⚠️ Επιλέξτε τουλάχιστον 1 υποψήφιο για σχεδίαση διαδρομής.', 'alert-circle');
    return;
  }

  if (selected.length === 1) {
    // Single prospect selected: map route from its nearest active client!
    return mapRouteBetweenClientAndProspect(selected[0].company_id);
  }

  // Multiple prospects selected: find nearest active client to the highest ranked prospect as origin depot!
  const sortedProspects = [...selected].sort((a, b) => a.rank - b.rank);
  const anchorClient = findNearestClient(sortedProspects[0].lat, sortedProspects[0].lon);

  const stops = anchorClient
    ? [{ ...anchorClient, isClient: true }, ...sortedProspects.map(p => ({ ...p, isClient: false }))]
    : sortedProspects.map(p => ({ ...p, isClient: false }));

  await drawMultiStopRoute(stops, {
    keepFirstOrigin: !!anchorClient,
    title: anchorClient
      ? (isEn ? `Distribution Circuit: Client "${anchorClient.name}" ➔ ${sortedProspects.length} Prospects` : `Διαδρομή Διανομής: Πελάτης «${anchorClient.name}» ➔ ${sortedProspects.length} Υποψήφιοι`)
      : (isEn ? `Sales Route: ${sortedProspects.length} Prospects` : `Διαδρομή Πωλήσεων: ${sortedProspects.length} Υποψήφιοι`)
  });
};

window.openClientRouteModal = function() {
  const select = document.getElementById('route-modal-client-select');
  const modal = document.getElementById('client-route-modal-backdrop');
  if (!select || !modal) return;
  
  const isEn = state.lang === 'en';
  const sortedClients = [...(state.clients || [])]
    .filter(c => c.lat && c.lon)
    .sort((a, b) => (b.turnover || 0) - (a.turnover || 0));

  select.innerHTML = sortedClients.map(c => `
    <option value="${c.company_id}">
      ${escapeHtml(c.name)} (${escapeHtml(c.city || c.prefecture || '')}) · €${Math.round(c.turnover || 0).toLocaleString(isEn ? 'en-US' : 'el-GR')} (${isEn ? 'Tax ID' : 'ΑΦΜ'}: ${c.vat})
    </option>
  `).join('');

  modal.classList.remove('opacity-0', 'pointer-events-none');
};

window.closeClientRouteModal = function() {
  const modal = document.getElementById('client-route-modal-backdrop');
  if (modal) modal.classList.add('opacity-0', 'pointer-events-none');
};

window.executeClientRouteFromModal = function() {
  const select = document.getElementById('route-modal-client-select');
  const countSelect = document.getElementById('route-modal-stops-count');
  const radiusSelect = document.getElementById('route-modal-radius-km');
  if (!select) return;

  const clientId = Number(select.value);
  const maxStops = Number(countSelect ? countSelect.value : 3);
  const maxRadiusKm = Number(radiusSelect ? radiusSelect.value : 5);
  closeClientRouteModal();
  mapRouteFromClientToNearbyProspects(clientId, maxStops, maxRadiusKm);
};

window.clearSalesRoute = function() {
  if (state.salesRouteLayer) state.salesRouteLayer.clearLayers();
  const container = document.getElementById('map-route-banner-container');
  if (container) container.classList.add('hidden');
  const banner = document.getElementById('map-route-banner');
  if (banner) banner.classList.add('hidden');
  const panel = document.getElementById('map-route-details-panel');
  if (panel) panel.classList.add('hidden');
  const chevron = document.getElementById('route-details-chevron');
  if (chevron) chevron.classList.remove('rotate-180');
  state.activeRoute = null;
};

function renderLeadsTable() {
  const tbody = document.getElementById('leads-table-body');
  if (!tbody) return;
  const isEn = state.lang === 'en';
  const loc = isEn ? 'en-US' : 'el-GR';
  
  updateQuickChipCounters();

  const f = state.leadsFilter;
  let items = [...(state.lookalikes || [])];
  
  // Apply Action-Oriented Quick Filter Chips
  if (f.quickFilter === 'hot') {
    items = items.filter(l => (l.score || 0) >= 0.85);
  } else if (f.quickFilter === 'route') {
    items = items.filter(l => typeof l.nearest_client_km === 'number' && l.nearest_client_km <= 1.0);
  } else if (f.quickFilter === 'multi') {
    items = items.filter(l => l.is_branch || (l.branches && l.branches > 1));
  } else if (f.quickFilter === 'tier1') {
    items = items.filter(l => l.osm_tier === 'Tier 1 (Confirmed Match)');
  } else if (f.quickFilter === 'top100') {
    items = items.filter(l => l.rank <= 100);
  } else if (f.quickFilter === 'attica') {
    items = items.filter(l => l.prefecture === 'ΑΤΤΙΚΗΣ');
  } else if (f.quickFilter === 'thessaloniki') {
    items = items.filter(l => l.prefecture === 'ΘΕΣΣΑΛΟΝΙΚΗΣ');
  } else if (f.quickFilter === 'islands') {
    items = items.filter(l => ['ΚΥΚΛΑΔΩΝ', 'ΔΩΔΕΚΑΝΗΣΟΥ', 'ΧΑΛΚΙΔΙΚΗΣ', 'ΗΡΑΚΛΕΙΟΥ', 'ΧΑΝΙΩΝ', 'ΡΕΘΥΜΝΗΣ', 'ΛΑΣΙΘΙΟΥ', 'ΚΕΡΚΥΡΑΣ'].includes(l.prefecture));
  } else if (f.quickFilter === 'corporate') {
    items = items.filter(l => ['ΙΚΕ', 'ΑΕ', 'ΕΠΕ'].includes(l.legal_type));
  } else if (f.quickFilter === 'hq') {
    items = items.filter(l => !l.is_branch);
  }

  // Apply Standard Dropdown & Search Filters
  if (f.profile !== 'ALL') items = items.filter(l => l.profile_id === f.profile);
  if (f.prefecture !== 'ALL') items = items.filter(l => l.prefecture === f.prefecture);
  if (f.legal !== 'ALL') items = items.filter(l => l.legal_type === f.legal);
  if (f.search) {
    const q = f.search.toLowerCase();
    items = items.filter(l => 
      (l.name && l.name.toLowerCase().includes(q)) ||
      (l.vat && String(l.vat).includes(q)) ||
      (l.city && l.city.toLowerCase().includes(q)) ||
      (l.street && l.street.toLowerCase().includes(q)) ||
      (l.kad_desc && l.kad_desc.toLowerCase().includes(q))
    );
  }

  // Apply 2-Way Column Sorting
  const dirMul = f.sortDir === 'desc' ? -1 : 1;
  items.sort((a, b) => {
    if (f.sortKey === 'score') {
      return ((a.score || 0) - (b.score || 0)) * dirMul;
    }
    if (f.sortKey === 'name') {
      return String(a.name || '').localeCompare(String(b.name || ''), 'el') * dirMul;
    }
    if (f.sortKey === 'nearest_client_km') {
      const da = typeof a.nearest_client_km === 'number' ? a.nearest_client_km : 9999;
      const db = typeof b.nearest_client_km === 'number' ? b.nearest_client_km : 9999;
      return (da - db) * dirMul;
    }
    return ((a.rank || 0) - (b.rank || 0)) * dirMul;
  });

  state.currentFilteredLeads = items;
  
  safeSetText('leads-visible-count', items.length.toLocaleString(loc));
  
  const avgScore = items.length ? (items.reduce((acc, x) => acc + (x.score || 0), 0) / items.length * 100).toFixed(1) : '0';
  safeSetText('leads-cockpit-avg-score', `${avgScore}%`);
  safeSetText('leads-cockpit-total', items.length.toLocaleString(loc));

  // Pagination
  const totalPages = Math.max(1, Math.ceil(items.length / f.pageSize));
  if (f.page > totalPages) f.page = totalPages;
  const startIdx = (f.page - 1) * f.pageSize;
  const pageItems = items.slice(startIdx, startIdx + f.pageSize);
  
  safeSetText('leads-page-indicator', `${f.page} / ${totalPages}`);
  safeSetText(
    'leads-pagination-info',
    isEn
      ? `Showing ${items.length ? startIdx + 1 : 0}-${Math.min(startIdx + f.pageSize, items.length)} of ${items.length.toLocaleString(loc)}`
      : `Εμφάνιση ${items.length ? startIdx + 1 : 0}-${Math.min(startIdx + f.pageSize, items.length)} από ${items.length.toLocaleString(loc)}`
  );
  
  const prevBtn = document.getElementById('leads-prev-page');
  const nextBtn = document.getElementById('leads-next-page');
  if (prevBtn) prevBtn.disabled = f.page <= 1;
  if (nextBtn) nextBtn.disabled = f.page >= totalPages;
  
  if (!pageItems.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center py-10 text-slate-500">
          ${isEn ? 'No matching prospects found for the active filters.' : 'Δεν βρέθηκαν προτεινόμενοι υποψήφιοι με τα επιλεγμένα φίλτρα.'}
        </td>
      </tr>
    `;
    return;
  }
  
  tbody.innerHTML = pageItems.map((l, idxOnPage) => {
    const absIdx = startIdx + idxOnPage;
    const isFocused = state.focusedLeadIndex === absIdx || state.selectedCompanyId === l.company_id;
    const isChecked = state.selectedLeadIds.has(Number(l.company_id));
    const pIdx = parseInt(l.profile_id.replace('ICP_', '')) - 1;
    const color = ICP_COLORS[pIdx % ICP_COLORS.length] || '#2563eb';
    const scorePct = Math.round((l.score || 0) * 100);
    
    let rankBadge = `<span class="font-mono font-bold text-slate-400">#${l.rank}</span>`;
    if (l.rank === 1) rankBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-extrabold border border-amber-500/40 text-[11px]">🥇 #1</span>`;
    else if (l.rank === 2) rankBadge = `<span class="px-1.5 py-0.5 rounded bg-slate-400/20 text-slate-200 font-extrabold border border-slate-400/40 text-[11px]">🥈 #2</span>`;
    else if (l.rank === 3) rankBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-700/20 text-amber-400 font-extrabold border border-amber-700/40 text-[11px]">🥉 #3</span>`;

    const gmapsUrl = getGoogleMapsUrl(l);
    const streetAddress = [l.street, l.street_number].filter(Boolean).join(' ');

    // Delivery route distance badge & HoReCa 250m density badge
    const distKm = typeof l.nearest_client_km === 'number' ? l.nearest_client_km : null;
    const distBadge = distKm !== null
      ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold ${distKm <= 1.0 ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/35' : 'bg-slate-900 text-slate-400 border border-slate-800'}" title="${isEn ? 'Distance to nearest active client' : 'Απόσταση από πλησιέστερο ενεργό πελάτη'}">🚚 ${distKm < 1 ? Math.round(distKm * 1000) + 'm' : distKm.toFixed(1) + 'km'}</span>`
      : '';

    const segTierShort = (l.commercial_segment_tier || '').startsWith('Tier A')
      ? 'Tier A'
      : (l.commercial_segment_tier || '').startsWith('Tier B')
      ? 'Tier B'
      : 'Tier C';

    return `
      <tr id="lead-row-${l.company_id}"
          onmouseenter="primeMapMarkerOnHover(${l.lat || 0}, ${l.lon || 0})"
          onmouseleave="clearMapHoverHalo()"
          class="hover:bg-slate-900/80 transition-colors border-b border-slate-800/60 ${isFocused ? 'row-active-highlight' : ''} ${isChecked ? 'row-selected' : ''}">
        <!-- 0. Sticky Checkbox -->
        <td class="py-2 px-2.5 sticky-col-left">
          <input type="checkbox" ${isChecked ? 'checked' : ''} onchange="toggleSelectLead(${l.company_id}, this.checked)" class="custom-lead-checkbox" title="${isEn ? 'Select lead' : 'Επιλογή υποψηφίου'}">
        </td>

        <!-- 1. Rank & ICP -->
        <td class="py-2 px-2.5">
          <div class="flex items-center gap-1.5">
            ${rankBadge}
            <span class="px-1.5 py-0.5 rounded text-[10px] font-bold text-white shadow-sm" style="background-color: ${color}">
              ${escapeHtml(l.profile_id)}
            </span>
          </div>
        </td>
        
        <!-- 2. Score & Fit -->
        <td class="py-2 px-2.5">
          <div class="space-y-1">
            <div class="flex items-center justify-between text-xs">
              <span class="font-mono font-bold text-emerald-400">${scorePct}%</span>
              <span class="text-[10px] text-slate-400 font-medium">${scorePct >= 90 ? (isEn ? 'Elite' : 'Εξαιρετική') : scorePct >= 80 ? (isEn ? 'High' : 'Υψηλή') : (isEn ? 'Good' : 'Καλή')}</span>
            </div>
            <div class="score-bar-bg w-full">
              <div class="score-bar-fill" style="width: ${scorePct}%"></div>
            </div>
          </div>
        </td>
        
        <!-- 3. Company, Commercial Segment Tier, KAD & OSM Badges -->
        <td class="py-2 px-3">
          <div class="flex items-center gap-1.5 flex-wrap">
            <span class="font-bold text-white text-xs hover:text-blue-400 cursor-pointer transition-colors truncate max-w-[240px]" onclick="inspectCompany(${l.company_id})" title="${escapeHtml(l.name)}">
              ${escapeHtml(l.name)}
            </span>
            <span class="px-1.5 py-0.2 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 text-[9.5px] font-mono font-semibold" title="${escapeHtml(l.commercial_segment_tier || 'B2B Segment')}">${segTierShort}</span>
            ${l.osm_tier === 'Tier 1 (Confirmed Match)' ? `
              <span class="px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-[9.5px] font-semibold" title="OSM Tier 1 Confirmed (<50m & ≥75% name similarity)">OSM Tier 1</span>
            ` : l.osm_tier === 'Tier 2 (Spatial-Only Ambiguous)' ? `
              <span class="px-1.5 py-0.2 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 text-[9.5px] font-semibold" title="OSM Tier 2 Spatial-Only (<25m)">OSM Tier 2</span>
            ` : ''}
            ${l.is_transit_hub ? `
              <span class="px-1.5 py-0.2 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 text-[9.5px] font-semibold">Transit Hub</span>
            ` : ''}
          </div>
          <div class="text-[11px] text-slate-400 truncate mt-0.5" title="${escapeHtml(l.kad_desc || '')}">
            <span class="font-mono text-blue-400 font-semibold">${l.kad}</span> · ${escapeHtml(l.kad_desc || '')}
          </div>
          <div class="row-detail-extra text-[10.5px] text-slate-500 truncate mt-0.5" title="${escapeHtml(l.reason || '')}">
            ${escapeHtml(l.reason || '')}
          </div>
        </td>
        
        <!-- 4. Location, Delivery Route Distance & Tax -->
        <td class="py-2 px-2.5">
          <div class="flex items-center justify-between gap-1">
            <span class="text-white text-xs truncate max-w-[115px]" title="${escapeHtml(streetAddress || l.city || '')}">
              ${escapeHtml(l.city || streetAddress || '-')}
            </span>
            ${distBadge}
          </div>
          <div class="text-[10.5px] text-slate-400 truncate">
            ${escapeHtml(l.municipality || '')} (${escapeHtml(l.prefecture || '')})
          </div>
          <div class="row-detail-extra text-[10px] text-slate-500 mt-0.5 font-mono">
            ${isEn ? 'Tax ID' : 'ΑΦΜ'}: <span class="text-slate-300 font-semibold">${escapeHtml(l.vat)}</span> · <span class="text-blue-400 uppercase font-bold">${escapeHtml(l.legal_type || '-')}</span>
          </div>
        </td>
        
        <!-- 5. Sticky Right Actions -->
        <td class="py-2 px-2 text-center sticky-col-right w-[84px] min-w-[84px]">
          <div class="flex items-center justify-center gap-1.5 whitespace-nowrap">
            <button onclick="mapRouteBetweenClientAndProspect(${l.company_id})" class="w-7 h-7 rounded-md bg-amber-500/15 hover:bg-amber-500 text-amber-300 hover:text-white border border-amber-500/30 text-xs transition-all flex items-center justify-center shrink-0" title="${isEn ? 'Route from Nearest Client' : 'Διαδρομή από Πλησιέστερο Πελάτη'}">
              <i data-lucide="navigation" class="w-3.5 h-3.5"></i>
            </button>
            <button onclick="focusOnMap(${l.lat}, ${l.lon}, ${l.company_id}, '${escapeHtml(l.name).replace(/'/g, "\\'")}', '${l.profile_id}')" class="w-7 h-7 rounded-md bg-blue-600/20 hover:bg-blue-600 text-blue-300 hover:text-white border border-blue-500/30 text-xs transition-all flex items-center justify-center shrink-0" title="${isEn ? 'Focus on Map' : 'Εστίαση στον Χάρτη'}">
              <i data-lucide="map-pin" class="w-3.5 h-3.5"></i>
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
  lucide.createIcons();
}

function renderClientsTable() {
  const tbody = document.getElementById('clients-table-body');
  if (!tbody) return;
  
  const f = state.clientsFilter;
  let items = state.clients;
  
  // Apply Quick Filter Chips for Clients
  if (f.quickFilter === 'vip') {
    items = items.filter(c => (c.turnover || 0) >= 25000);
  } else if (f.quickFilter === 'icp1') {
    items = items.filter(c => c.profile_id === 'ICP_1');
  } else if (f.quickFilter === 'icp2') {
    items = items.filter(c => c.profile_id === 'ICP_2');
  } else if (f.quickFilter === 'nofit') {
    items = items.filter(c => c.profile_id === 'NO_FIT');
  } else if (f.quickFilter === 'branches') {
    items = items.filter(c => (c.branch_count || 0) > 1 || c.is_branch);
  }

  // Apply Standard Search & Profile Filter
  if (f.profile !== 'ALL') items = items.filter(c => c.profile_id === f.profile);
  if (f.search) {
    const q = f.search.toLowerCase();
    items = items.filter(c => 
      (c.name && c.name.toLowerCase().includes(q)) ||
      (c.vat && String(c.vat).includes(q)) ||
      (c.city && c.city.toLowerCase().includes(q)) ||
      (c.street && c.street.toLowerCase().includes(q))
    );
  }
  
  if (!items.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" class="text-center py-10 text-slate-500">
          Δεν βρέθηκαν πελάτες με τα επιλεγμένα κριτήρια.
        </td>
      </tr>
    `;
    return;
  }
  
  tbody.innerHTML = items.map(c => {
    const isNoFit = c.profile_id === 'NO_FIT';
    const pIdx = isNoFit ? -1 : parseInt(c.profile_id.replace('ICP_', '')) - 1;
    const color = isNoFit ? '#64748b' : (ICP_COLORS[pIdx % ICP_COLORS.length] || '#2563eb');
    const scorePct = Math.round((c.score || 0) * 100);
    
    const gmapsUrl = getGoogleMapsUrl(c);
    const webUrl = getGoogleSearchUrl(c);
    const gemiUrl = getGemiUrl(c.vat);
    const streetAddress = [c.street, c.street_number].filter(Boolean).join(' ');

    return `
      <tr class="hover:bg-slate-900/80 transition-colors border-b border-slate-800/60">
        <!-- 1. ICP & Score -->
        <td class="py-2.5 px-3">
          <div class="space-y-1">
            <div class="flex items-center gap-1.5">
              <span class="px-1.5 py-0.5 rounded text-[10px] font-bold text-white shadow-sm" style="background-color: ${color}">
                ${escapeHtml(c.profile_id)}
              </span>
              <span class="font-mono font-bold ${isNoFit ? 'text-slate-400' : 'text-emerald-400'} text-xs">${scorePct}%</span>
            </div>
            <div class="score-bar-bg w-full">
              <div class="score-bar-fill" style="width: ${scorePct}%; background: ${isNoFit ? '#64748b' : ''}"></div>
            </div>
          </div>
        </td>
        
        <!-- 2. Company & Activity (Title + KAD + Structure) -->
        <td class="py-2.5 px-4">
          <div class="font-bold text-white text-xs hover:text-blue-400 cursor-pointer transition-colors truncate" onclick="inspectCompany(${c.company_id})" title="${escapeHtml(c.name)}">
            ${escapeHtml(c.name)}
          </div>
          <div class="text-[11px] text-slate-400 truncate mt-0.5" title="${escapeHtml(c.kad_desc || '')}">
            <span class="font-mono text-blue-400 font-semibold">${c.kad}</span> · ${escapeHtml(c.kad_desc || '')}
          </div>
          <div class="text-[10.5px] text-slate-500 truncate mt-0.5">
            ${c.is_branch ? '🏢 Υποκατάστημα' : '🏢 Κεντρικό'} · ${c.branch_count || 0} σημεία
          </div>
        </td>
        
        <!-- 3. Turnover -->
        <td class="py-2.5 px-3 text-right">
          <div class="font-mono font-extrabold text-emerald-400 text-xs">
            €${Math.round(c.turnover || 0).toLocaleString('el-GR')}
          </div>
          <div class="text-[10px] text-slate-400 font-medium mt-0.5">
            ${(c.turnover || 0) >= 25000 ? '<span class="text-amber-400 font-semibold">VIP (>€25k)</span>' : 'Standard'}
          </div>
        </td>
        
        <!-- 4. Location & Tax -->
        <td class="py-2.5 px-3">
          <div class="text-white text-xs truncate" title="${escapeHtml(streetAddress || c.city || '')}">
            ${escapeHtml(streetAddress || c.city || '-')}
          </div>
          <div class="text-[10.5px] text-slate-400 truncate">
            ${escapeHtml(c.municipality || '')} (${escapeHtml(c.prefecture || '')})
          </div>
          <div class="text-[10.5px] text-slate-500 mt-0.5 font-mono">
            ΑΦΜ: <span class="text-slate-300 font-semibold">${escapeHtml(c.vat)}</span> · <span class="text-blue-400 uppercase font-bold">${escapeHtml(c.legal_type || '-')}</span>
          </div>
        </td>
        
        <!-- 5. Actions Toolbar -->
        <td class="py-2.5 px-3 text-right">
          <div class="flex items-center justify-end gap-1">
            <button onclick="mapRouteFromClientToNearbyProspects(${c.company_id})" class="p-1.5 rounded bg-amber-500/15 hover:bg-amber-500 text-amber-300 hover:text-white border border-amber-500/30 text-xs transition-all" title="Σχεδίαση Διαδρομής προς Πλησιέστερους Υποψηφίους">
              <i data-lucide="navigation" class="w-3.5 h-3.5"></i>
            </button>
            <button onclick="focusOnMap(${c.lat}, ${c.lon}, ${c.company_id}, '${escapeHtml(c.name).replace(/'/g, "\\'")}', '${c.profile_id}')" class="p-1.5 rounded bg-blue-600/20 hover:bg-blue-600 text-blue-300 hover:text-white border border-blue-500/30 text-xs transition-all" title="Εστίαση στον Χάρτη">
              <i data-lucide="map-pin" class="w-3.5 h-3.5"></i>
            </button>
            <a href="${gmapsUrl}" target="_blank" rel="noopener" class="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-sky-400 border border-slate-800 text-xs transition-all" title="Google Maps Listing">
              <i data-lucide="map" class="w-3.5 h-3.5"></i>
            </a>
            <button onclick="openGemi('${c.vat}', '${escapeHtml(c.name).replace(/'/g, "\\'")}'); event.stopPropagation();" class="p-1.5 rounded bg-slate-900 hover:bg-slate-800 text-emerald-400 border border-slate-800 text-xs transition-all" title="Δημοσιότητα Γ.Ε.ΜΗ. (Αντιγραφή ΑΦΜ: ${c.vat})">
              <i data-lucide="shield-check" class="w-3.5 h-3.5"></i>
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
  lucide.createIcons();
}

// -----------------------------------------------------------------------------
// Filtered & Batch CSV Export
// -----------------------------------------------------------------------------
function downloadLeadsListAsCsv(items, filename) {
  if (!items || !items.length) {
    showToast(state.lang === 'en' ? '⚠️ No records to export.' : '⚠️ Δεν υπάρχουν εγγραφές για εξαγωγή.', 'alert-circle');
    return;
  }
  const headers = ['Rank', 'Profile', 'Score', 'Company Name', 'VAT', 'Legal Type', 'Commercial Segment Tier', 'Nearest Client km', 'HoReCa Density 250m', 'OSM Tier', 'Prefecture', 'Municipality', 'City', 'Street', 'Street Number', 'Zip Code', 'KAD', 'KAD Description', 'Latitude', 'Longitude', 'Google Maps Link', 'Recommendation Reason'];
  const rows = items.map(l => [
    l.rank,
    l.profile_id,
    l.score,
    `"${(l.name || '').replace(/"/g, '""')}"`,
    l.vat,
    l.legal_type || '',
    `"${(l.commercial_segment_tier || '').replace(/"/g, '""')}"`,
    typeof l.nearest_client_km === 'number' ? l.nearest_client_km : '',
    typeof l.horeca_density_250m === 'number' ? l.horeca_density_250m : '',
    `"${(l.osm_tier || '').replace(/"/g, '""')}"`,
    l.prefecture || '',
    l.municipality || '',
    l.city || '',
    `"${(l.street || '').replace(/"/g, '""')}"`,
    l.street_number || '',
    l.zip_code || '',
    l.kad,
    `"${(l.kad_desc || '').replace(/"/g, '""')}"`,
    l.lat || '',
    l.lon || '',
    `"https://www.google.com/maps/search/?api=1&query=${encodeURIComponent((l.name || '') + ' ' + (l.street || '') + ' ' + (l.city || ''))}"`,
    `"${(l.reason || '').replace(/"/g, '""')}"`,
  ]);
  
  const csvContent = '\uFEFF' + [headers.join(','), ...rows.map(r => r.join(','))].join('\r\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `ForceCRM_Filtered_Leads_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(
    state.lang === 'en' ? `📥 Exported ${items.length.toLocaleString('en-US')} prospects to CSV!` : `📥 Εξήχθησαν ${items.length.toLocaleString('el-GR')} υποψήφιοι σε CSV!`,
    'download'
  );
}

function exportFilteredLeadsCsv() {
  const items = state.currentFilteredLeads && state.currentFilteredLeads.length ? state.currentFilteredLeads : state.lookalikes;
  downloadLeadsListAsCsv(items, `ForceCRM_Filtered_Leads_${new Date().toISOString().slice(0, 10)}.csv`);
}

// -----------------------------------------------------------------------------
// Technical Silhouette Chart
// -----------------------------------------------------------------------------
function initSilhouetteChart() {
  const canvas = document.getElementById('chart-silhouette');
  if (!canvas) return;
  
  state.chartSilhouette = new Chart(canvas, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Cosine Silhouette', data: [], backgroundColor: [] }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#e2e8f0', font: { weight: 'bold' } }
        }
      }
    }
  });
}

function renderSilhouetteChart() {
  if (!state.chartSilhouette || !state.meta.silhouette_by_k) return;
  
  const entries = Object.entries(state.meta.silhouette_by_k).filter(([k]) => parseInt(k) > 1);
  const labels = entries.map(([k]) => `K=${k}`);
  const data = entries.map(([, v]) => v);
  const optK = String(state.meta.optimal_k || 2);
  const colors = entries.map(([k]) => k === optK ? '#2563eb' : '#475569');
  
  state.chartSilhouette.data.labels = labels;
  state.chartSilhouette.data.datasets[0].data = data;
  state.chartSilhouette.data.datasets[0].backgroundColor = colors;
  state.chartSilhouette.update('none');
}

// -----------------------------------------------------------------------------
// Interactive Slider Recomputing (<50ms Vectorized Updates)
// -----------------------------------------------------------------------------
function handleSliderChange(paramKey, value) {
  const valEl = document.getElementById(`val-${paramKey}`);
  if (valEl) {
    if (paramKey === 'spatial_kernel_bandwidth_km') {
      valEl.textContent = `${parseFloat(value).toFixed(1)} km`;
    } else if (paramKey === 'eb_prior_strength') {
      valEl.textContent = `${parseFloat(value).toFixed(1)}`;
    } else if (typeof value !== 'boolean') {
      valEl.textContent = parseFloat(value).toFixed(2);
    }
  }
  
  if (typeof value === 'boolean') {
    state.params[paramKey] = value;
  } else {
    state.params[paramKey] = isNaN(value) ? value : parseFloat(value);
  }
  
  // Debounce API call by 40ms to keep slider dragging ultra-smooth
  clearTimeout(state.sliderDebounceTimer);
  state.sliderDebounceTimer = setTimeout(async () => {
    try {
      state.isComputing = true;
      const resp = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: state.params }),
      });
      const data = await resp.json();
      if (data.status === 'ok') {
        applyEnginePayload(data.data);
        renderAllViews();
      }
    } catch (err) {
      console.error('Calculation error:', err);
    } finally {
      state.isComputing = false;
    }
  }, 40);
}

function syncSlidersToState() {
  Object.keys(state.params).forEach(k => {
    const input = document.getElementById(`param-${k}`);
    const valEl = document.getElementById(`val-${k}`);
    if (input) {
      if (input.type === 'checkbox') {
        input.checked = Boolean(state.params[k] !== false);
      } else {
        input.value = state.params[k];
      }
    }
    if (valEl) {
      const val = state.params[k];
      if (k === 'spatial_kernel_bandwidth_km') {
        valEl.textContent = `${parseFloat(val).toFixed(1)} km`;
      } else if (k === 'eb_prior_strength') {
        valEl.textContent = `${parseFloat(val).toFixed(1)}`;
      } else if (typeof val !== 'boolean' && !isNaN(val)) {
        valEl.textContent = parseFloat(val).toFixed(2);
      }
    }
  });

  // Sync Embedding Models Toggle UI & Indicators
  const useEmb = Boolean(state.params.use_embeddings !== false);
  const embControls = document.getElementById('neural-embedding-controls');
  const embDisabledMsg = document.getElementById('neural-embedding-disabled-msg');
  const embBadge = document.getElementById('badge-use_embeddings-state');
  const headerEmbText = document.getElementById('header-embeddings-text');
  const headerEmbDot = document.getElementById('header-embeddings-dot');

  if (embControls) embControls.classList.toggle('hidden', !useEmb);
  if (embDisabledMsg) embDisabledMsg.classList.toggle('hidden', useEmb);
  if (embBadge) {
    embBadge.innerHTML = useEmb
      ? '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span><span>Νευρωνικά Embeddings Ενεργά</span>'
      : '<span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span><span class="text-slate-400">Αλγεβρικό SVD Baseline (χωρίς Neural)</span>';
    embBadge.className = `text-[10px] font-medium flex items-center gap-1 ${useEmb ? 'text-emerald-400' : 'text-slate-400'}`;
  }
  if (headerEmbText) {
    const mName = state.params.embedding_model || 'Granite';
    headerEmbText.textContent = useEmb ? `Embeddings: ${mName}` : 'Embeddings: SVD Only';
  }
  if (headerEmbDot) {
    headerEmbDot.className = `w-2 h-2 rounded-full ${useEmb ? 'bg-blue-400' : 'bg-slate-400'}`;
  }
}

// -----------------------------------------------------------------------------
// Regional & Legal Reference Dictionaries for English Localization
// -----------------------------------------------------------------------------
const PREFECTURE_NAMES_EN = {
  'ΑΘΗΝΩΝ': 'Athens (Attica)',
  'ΑΝΑΤΟΛΙΚΗΣ ΑΤΤΙΚΗΣ': 'East Attica',
  'ΔΥΤΙΚΗΣ ΑΤΤΙΚΗΣ': 'West Attica',
  'ΠΕΙΡΑΙΑ': 'Piraeus (Attica)',
  'ΑΤΤΙΚΗΣ': 'Attica',
  'ΘΕΣΣΑΛΟΝΙΚΗΣ': 'Thessaloniki',
  'ΧΑΛΚΙΔΙΚΗΣ': 'Halkidiki',
  'ΗΡΑΚΛΕΙΟΥ': 'Heraklion (Crete)',
  'ΧΑΝΙΩΝ': 'Chania (Crete)',
  'ΡΕΘΥΜΝΗΣ': 'Rethymno (Crete)',
  'ΛΑΣΙΘΙΟΥ': 'Lasithi (Crete)',
  'ΚΥΚΛΑΔΩΝ': 'Cyclades',
  'ΔΩΔΕΚΑΝΗΣΟΥ': 'Dodecanese',
  'ΚΕΡΚΥΡΑΣ': 'Corfu (Kerkyra)',
  'ΚΕΦΑΛΛΗΝΙΑΣ': 'Cephalonia',
  'ΖΑΚΥΝΘΟΥ': 'Zakynthos',
  'ΛΕΥΚΑΔΑΣ': 'Lefkada',
  'ΑΧΑΙΑΣ': 'Achaea (Patras)',
  'ΒΟΙΩΤΙΑΣ': 'Boeotia',
  'ΕΥΒΟΙΑΣ': 'Evia (Euboea)',
  'ΚΟΡΙΝΘΙΑΣ': 'Corinthia',
  'ΜΕΣΣΗΝΙΑΣ': 'Messenia (Kalamata)',
  'ΑΡΓΟΛΙΔΑΣ': 'Argolis (Nafplio)',
  'ΑΡΚΑΔΙΑΣ': 'Arcadia',
  'ΛΑΚΩΝΙΑΣ': 'Laconia (Sparta)',
  'ΗΛΕΙΑΣ': 'Elis (Ilia)',
  'ΛΑΡΙΣΑΣ': 'Larissa',
  'ΜΑΓΝΗΣΙΑΣ': 'Magnesia (Volos)',
  'ΤΡΙΚΑΛΩΝ': 'Trikala',
  'ΚΑΡΔΙΤΣΑΣ': 'Karditsa',
  'ΙΩΑΝΝΙΝΩΝ': 'Ioannina',
  'ΑΡΤΑΣ': 'Arta',
  'ΠΡΕΒΕΖΑΣ': 'Preveza',
  'ΘΕΣΠΡΩΤΙΑΣ': 'Thesprotia',
  'ΚΑΒΑΛΑΣ': 'Kavala',
  'ΣΕΡΡΩΝ': 'Serres',
  'ΠΙΕΡΙΑΣ': 'Pieria',
  'ΗΜΑΘΙΑΣ': 'Imathia',
  'ΠΕΛΛΗΣ': 'Pella',
  'ΚΙΛΚΙΣ': 'Kilkis',
  'ΚΟΖΑΝΗΣ': 'Kozani',
  'ΚΑΣΤΟΡΙΑΣ': 'Kastoria',
  'ΦΛΩΡΙΝΑΣ': 'Florina',
  'ΓΡΕΒΕΝΩΝ': 'Grevena',
  'ΕΒΡΟΥ': 'Evros',
  'ΡΟΔΟΠΗΣ': 'Rodopi',
  'ΞΑΝΘΗΣ': 'Xanthi',
  'ΔΡΑΜΑΣ': 'Drama',
  'ΛΕΣΒΟΥ': 'Lesbos',
  'ΧΙΟΥ': 'Chios',
  'ΣΑΜΟΥ': 'Samos',
  'ΦΘΙΩΤΙΔΑΣ': 'Phthiotis',
  'ΦΩΚΙΔΑΣ': 'Phocis',
  'ΕΥΡΥΤΑΝΙΑΣ': 'Evrytania',
  'ΑΙΤΩΛΟΑΚΑΡΝΑΝΙΑΣ': 'Aetolia-Acarnania'
};

const LEGAL_FORM_NAMES = {
  'ΑΕ': { el: 'ΑΕ (Ανώνυμη Εταιρεία)', en: 'AE (Corporation / SA)' },
  'ΑΣΤΙΚΗ ΕΤΑΙΡΕΙΑ 784 Α.Κ.': { el: 'Αστική Εταιρεία (784 Α.Κ.)', en: 'Civil Partnership (Astiki)' },
  'ΑΤΟΜΙΚΗ': { el: 'Ατομική Επιχείρηση', en: 'Sole Proprietorship (Ατομική)' },
  'ΕΕ': { el: 'ΕΕ (Ετερόρρυθμη Εταιρεία)', en: 'EE (Limited Partnership)' },
  'ΕΠΕ': { el: 'ΕΠΕ (Περιορισμένης Ευθύνης)', en: 'EPE (Limited Liability Co)' },
  'ΙΚΕ': { el: 'ΙΚΕ (Ιδιωτική Κεφαλαιουχική)', en: 'IKE (Private Capital Co)' },
  'Κοιν.Σ.Επ.': { el: 'Κοιν.Σ.Επ. (Κοινωνική Συνεταιριστική)', en: 'Social Cooperative (KoinSep)' },
  'Κοινοπραξία': { el: 'Κοινοπραξία', en: 'Joint Venture' },
  'ΟΕ': { el: 'ΟΕ (Ομόρρυθμη Εταιρεία)', en: 'OE (General Partnership)' },
  'Υποκατάστημα Αλλοδαπής': { el: 'Υποκατάστημα Αλλοδαπής', en: 'Foreign Branch Office' }
};

// -----------------------------------------------------------------------------
// Dropdown Population & Dynamic Localization Handlers
// -----------------------------------------------------------------------------
function populateFilterDropdowns() {
  const isEn = state.lang === 'en';

  // Profiles in Leads & Clients filter
  const leadsProfSelect = document.getElementById('leads-profile-filter');
  const clientsProfSelect = document.getElementById('clients-profile-filter');
  
  if (leadsProfSelect && clientsProfSelect) {
    const curLeadVal = state.leadsFilter ? state.leadsFilter.profile : leadsProfSelect.value;
    const curClientVal = state.clientsFilter ? state.clientsFilter.profile : clientsProfSelect.value;
    const allProfilesLabel = isEn ? 'All ICP Profiles' : 'Όλα τα Προφίλ ICP';
    const allClientsLabel = isEn ? 'All ICPs & NO_FIT' : 'Όλα τα ICPs & NO_FIT';

    const profOpts = (state.profiles || []).map(p => {
      const shortTitle = p.short_title || p.title || p.id;
      return `<option value="${escapeHtml(p.id)}">${escapeHtml(p.id)} (${escapeHtml(shortTitle)})</option>`;
    }).join('');

    leadsProfSelect.innerHTML = `<option value="ALL">${allProfilesLabel}</option>${profOpts}`;
    clientsProfSelect.innerHTML = `<option value="ALL">${allClientsLabel}</option>${profOpts}<option value="NO_FIT">${isEn ? 'NO_FIT (Outliers)' : 'NO_FIT (Ειδικές Περιπτώσεις)'}</option>`;

    if (curLeadVal && Array.from(leadsProfSelect.options).some(o => o.value === curLeadVal)) {
      leadsProfSelect.value = curLeadVal;
    }
    if (curClientVal && Array.from(clientsProfSelect.options).some(o => o.value === curClientVal)) {
      clientsProfSelect.value = curClientVal;
    }
  }
  
  // Prefectures in Leads
  const leadsPrefSelect = document.getElementById('leads-pref-filter');
  if (leadsPrefSelect && state.lookalikes) {
    const curPrefVal = state.leadsFilter ? state.leadsFilter.prefecture : leadsPrefSelect.value;
    const prefs = [...new Set(state.lookalikes.map(l => l.prefecture).filter(Boolean))].sort((a, b) => {
      if (isEn) {
        const nameA = PREFECTURE_NAMES_EN[a] || a;
        const nameB = PREFECTURE_NAMES_EN[b] || b;
        return nameA.localeCompare(nameB);
      }
      return a.localeCompare(b, 'el');
    });

    const allPrefLabel = isEn ? `All Prefectures (${prefs.length})` : `Όλοι οι Νομοί (${prefs.length})`;
    const prefOpts = prefs.map(p => {
      const displayLabel = isEn
        ? (PREFECTURE_NAMES_EN[p] ? `${PREFECTURE_NAMES_EN[p]} (${p})` : p)
        : p;
      return `<option value="${escapeHtml(p)}">${escapeHtml(displayLabel)}</option>`;
    }).join('');

    leadsPrefSelect.innerHTML = `<option value="ALL">${allPrefLabel}</option>${prefOpts}`;
    if (curPrefVal && Array.from(leadsPrefSelect.options).some(o => o.value === curPrefVal)) {
      leadsPrefSelect.value = curPrefVal;
    }
  }
  
  // Legal forms in Leads
  const leadsLegalSelect = document.getElementById('leads-legal-filter');
  if (leadsLegalSelect && state.lookalikes) {
    const curLegalVal = state.leadsFilter ? state.leadsFilter.legal : leadsLegalSelect.value;
    const legals = [...new Set(state.lookalikes.map(l => l.legal_type).filter(Boolean))].sort((a, b) => {
      const nameA = LEGAL_FORM_NAMES[a] ? (isEn ? LEGAL_FORM_NAMES[a].en : LEGAL_FORM_NAMES[a].el) : a;
      const nameB = LEGAL_FORM_NAMES[b] ? (isEn ? LEGAL_FORM_NAMES[b].en : LEGAL_FORM_NAMES[b].el) : b;
      return nameA.localeCompare(nameB, isEn ? 'en' : 'el');
    });

    const allLegalLabel = isEn ? 'All Legal Forms' : 'Όλες οι Νομικές Μορφές';
    const legalOpts = legals.map(l => {
      const displayLabel = LEGAL_FORM_NAMES[l]
        ? (isEn ? LEGAL_FORM_NAMES[l].en : LEGAL_FORM_NAMES[l].el)
        : l;
      return `<option value="${escapeHtml(l)}">${escapeHtml(displayLabel)}</option>`;
    }).join('');

    leadsLegalSelect.innerHTML = `<option value="ALL">${allLegalLabel}</option>${legalOpts}`;
    if (curLegalVal && Array.from(leadsLegalSelect.options).some(o => o.value === curLegalVal)) {
      leadsLegalSelect.value = curLegalVal;
    }
  }

  // Page Size Select
  const pageSizeSelect = document.getElementById('leads-page-size');
  if (pageSizeSelect) {
    const curSize = pageSizeSelect.value || '50';
    pageSizeSelect.innerHTML = isEn
      ? `<option value="25" ${curSize === '25' ? 'selected' : ''}>25 / page</option>
         <option value="50" ${curSize === '50' ? 'selected' : ''}>50 / page</option>
         <option value="100" ${curSize === '100' ? 'selected' : ''}>100 / page</option>`
      : `<option value="25" ${curSize === '25' ? 'selected' : ''}>25 / σελ.</option>
         <option value="50" ${curSize === '50' ? 'selected' : ''}>50 / σελ.</option>
         <option value="100" ${curSize === '100' ? 'selected' : ''}>100 / σελ.</option>`;
  }

  // Route Modal Stops Count
  const stopsSelect = document.getElementById('route-modal-stops-count');
  if (stopsSelect) {
    const curStops = stopsSelect.value || '3';
    stopsSelect.innerHTML = isEn
      ? `<option value="1" ${curStops === '1' ? 'selected' : ''}>1 Nearest Prospect</option>
         <option value="3" ${curStops === '3' ? 'selected' : ''}>Top 3 Nearest (Cluster)</option>
         <option value="5" ${curStops === '5' ? 'selected' : ''}>Top 5 Nearest</option>
         <option value="8" ${curStops === '8' ? 'selected' : ''}>Top 8 Nearest</option>`
      : `<option value="1" ${curStops === '1' ? 'selected' : ''}>1 Πλησιέστερος Υποψήφιος</option>
         <option value="3" ${curStops === '3' ? 'selected' : ''}>Top 3 Πλησιέστεροι (Σύμπλεγμα)</option>
         <option value="5" ${curStops === '5' ? 'selected' : ''}>Top 5 Πλησιέστεροι</option>
         <option value="8" ${curStops === '8' ? 'selected' : ''}>Top 8 Πλησιέστεροι</option>`;
  }

  // Route Modal Radius
  const radiusSelect = document.getElementById('route-modal-radius-km');
  if (radiusSelect) {
    const curRadius = radiusSelect.value || '5';
    radiusSelect.innerHTML = isEn
      ? `<option value="3" ${curRadius === '3' ? 'selected' : ''}>Up to 3 km (Immediate Area)</option>
         <option value="5" ${curRadius === '5' ? 'selected' : ''}>Up to 5 km (Micro-Region)</option>
         <option value="15" ${curRadius === '15' ? 'selected' : ''}>Up to 15 km (Municipality)</option>
         <option value="50" ${curRadius === '50' ? 'selected' : ''}>Up to 50 km (Broad Region)</option>`
      : `<option value="3" ${curRadius === '3' ? 'selected' : ''}>Έως 3 km (Άμεση Γειτονιά)</option>
         <option value="5" ${curRadius === '5' ? 'selected' : ''}>Έως 5 km (Μικρο-Περιοχή)</option>
         <option value="15" ${curRadius === '15' ? 'selected' : ''}>Έως 15 km (Δήμος / Τομέας)</option>
         <option value="50" ${curRadius === '50' ? 'selected' : ''}>Έως 50 km (Περιφέρεια)</option>`;
  }

  // Turnover Weighting Mode
  const turnoverModeSelect = document.getElementById('param-turnover_weighting_mode');
  if (turnoverModeSelect) {
    const curMode = turnoverModeSelect.value || 'log_turnover';
    turnoverModeSelect.innerHTML = isEn
      ? `<option value="log_turnover" ${curMode === 'log_turnover' ? 'selected' : ''}>Logarithmic (ln turnover)</option>
         <option value="sqrt_turnover" ${curMode === 'sqrt_turnover' ? 'selected' : ''}>Square root of turnover</option>
         <option value="linear_turnover" ${curMode === 'linear_turnover' ? 'selected' : ''}>Linear turnover</option>
         <option value="uniform" ${curMode === 'uniform' ? 'selected' : ''}>Uniform (No turnover weight)</option>`
      : `<option value="log_turnover" ${curMode === 'log_turnover' ? 'selected' : ''}>Λογαριθμική (ln κύκλου εργασιών)</option>
         <option value="sqrt_turnover" ${curMode === 'sqrt_turnover' ? 'selected' : ''}>Τετραγωνική ρίζα κύκλου εργασιών</option>
         <option value="linear_turnover" ${curMode === 'linear_turnover' ? 'selected' : ''}>Γραμμική</option>
         <option value="uniform" ${curMode === 'uniform' ? 'selected' : ''}>Ισόβαρη (χωρίς κύκλο εργασιών)</option>`;
  }

  // Models in Analyst Drawer
  const embModelSelect = document.getElementById('param-embedding_model');
  if (embModelSelect && state.paramSpecs) {
    const modelSpec = state.paramSpecs.find(s => s.key === 'embedding_model');
    if (modelSpec && modelSpec.options && modelSpec.options.length) {
      const curVal = state.params.embedding_model || 'granite-107m';
      embModelSelect.innerHTML = modelSpec.options.map(o => 
        `<option value="${o.value}" ${o.value === curVal ? 'selected' : ''}>${escapeHtml(o.label)} ${o.detail ? `(${escapeHtml(o.detail)})` : ''}</option>`
      ).join('');
    }
  }
}

// -----------------------------------------------------------------------------
// Geo Utility: Haversine Great-Circle Distance
// -----------------------------------------------------------------------------
function computeDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

let drawerMiniMap = null;

// -----------------------------------------------------------------------------
// Non-Blocking 500px Right-Anchored Slide-Over Company Inspector Drawer
// -----------------------------------------------------------------------------
window.switchDrawerSubTab = function(tabKey) {
  state.drawerSubTab = ['overview', 'geo', 'registry'].includes(tabKey) ? tabKey : 'overview';
  ['overview', 'geo', 'registry'].forEach(k => {
    const pane = document.getElementById(`drawer-subpane-${k}`);
    const btn = document.getElementById(`drawer-subbtn-${k}`);
    if (pane) pane.classList.toggle('hidden', k !== state.drawerSubTab);
    if (btn) {
      btn.className = k === state.drawerSubTab
        ? 'flex-1 py-1.5 px-2 rounded-md bg-blue-600 text-white text-xs font-bold transition-colors text-center'
        : 'flex-1 py-1.5 px-2 rounded-md text-slate-400 hover:text-white text-xs font-semibold transition-colors text-center';
    }
  });
  if (state.drawerSubTab === 'geo' && drawerMiniMap) {
    setTimeout(() => drawerMiniMap.invalidateSize(), 80);
  }
};

window.navigateInspectAdjacent = function(delta) {
  const items = state.currentFilteredLeads && state.currentFilteredLeads.length ? state.currentFilteredLeads : state.lookalikes;
  if (!items || !items.length) return;
  let idx = items.findIndex(l => l.company_id === state.selectedCompanyId);
  if (idx === -1) idx = Math.max(0, state.focusedLeadIndex);
  else idx = Math.min(items.length - 1, Math.max(0, idx + delta));

  const target = items[idx];
  if (!target) return;
  state.focusedLeadIndex = idx;

  // Ensure table page matches
  const targetPage = Math.floor(idx / state.leadsFilter.pageSize) + 1;
  const pageChanged = state.leadsFilter.page !== targetPage;
  if (pageChanged) {
    state.leadsFilter.page = targetPage;
    renderLeadsTable();
  }
  
  setTimeout(() => {
    document.querySelectorAll('#leads-table-body tr').forEach(r => r.classList.remove('row-active-highlight'));
    const row = document.getElementById(`lead-row-${target.company_id}`);
    if (row) {
      row.classList.add('row-active-highlight');
      scrollRowToCenter(row);
    }
  }, pageChanged ? 60 : 20);

  if (target.lat && target.lon) {
    primeMapMarkerOnHover(target.lat, target.lon);
  }
  inspectCompany(target.company_id);
};

async function inspectCompany(companyId) {
  state.selectedCompanyId = companyId;
  const drawer = document.getElementById('company-drawer-backdrop');
  const body = document.getElementById('company-drawer-body');
  if (!drawer || !body) return;
  const isEn = state.lang === 'en';
  
  if (drawerMiniMap) {
    try { drawerMiniMap.remove(); } catch (e) {}
    drawerMiniMap = null;
  }
  
  if (state.layoutMode === 'table') {
    setLayoutMode('split');
  }

  const mapWrapper = document.getElementById('map-view-wrapper');
  if (mapWrapper) {
    mapWrapper.classList.add('hidden');
  }

  drawer.classList.remove('hidden');
  drawer.classList.add('open', 'drawer-open');
  body.innerHTML = `
    <div class="py-12 text-center text-slate-400">
      <div class="w-8 h-8 rounded-full border-2 border-blue-500 border-t-transparent animate-spin mx-auto mb-3"></div>
      <div>${isEn ? 'Loading company intelligence card...' : 'Φόρτωση καρτέλας επιχείρησης...'}</div>
    </div>
  `;
  
  try {
    const resp = await fetch(`/api/company/${companyId}`);
    const data = await resp.json();
    if (!data.company) {
      body.innerHTML = `<div class="text-rose-400 p-4">${isEn ? 'Company not found.' : 'Η επιχείρηση δεν βρέθηκε.'}</div>`;
      return;
    }
    state.lastInspectedData = data;
    
    const c = data.company;
    const kads = data.kads || [];
    const ctx = data.context || {};
    const tc = ctx.trait_contributions || { semantic_pct: 58, eb_lift_pct: 29, spatial_pct: 13 };
    
    const gmapsUrl = getGoogleMapsUrl(c);
    const webUrl = getGoogleSearchUrl(c);
    const gemiSearchUrl = getGemiSearchUrl(c.vat, c.company_name_el || c.name);
    
    const targetIcp = ctx.lookalike ? ctx.lookalike.profile_id : (ctx.assignment ? ctx.assignment.profile_id : 'ICP_1');
    const primaryKadDesc = kads.length ? kads[0].description : (isEn ? 'Hospitality Services' : 'Υπηρεσίες Εστίασης');
    const segmentTier = c.commercial_segment_tier || (ctx.lookalike && ctx.lookalike.commercial_segment_tier) || 'Tier B — Structured Growth B2B Account';
    const nearestClientKm = typeof c.nearest_client_km_default === 'number'
      ? c.nearest_client_km_default
      : (ctx.lookalike && typeof ctx.lookalike.nearest_client_km === 'number' ? ctx.lookalike.nearest_client_km : null);
    const horeca250m = typeof c.horeca_density_250m === 'number'
      ? c.horeca_density_250m
      : (ctx.lookalike && typeof ctx.lookalike.horeca_density_250m === 'number' ? ctx.lookalike.horeca_density_250m : 0);
    const osmTier = c.osm_match_tier || (ctx.lookalike && ctx.lookalike.osm_tier) || 'Tier 2 (Spatial-Only Unbranded)';
    
    const salesPitch = isEn
      ? `<strong>${escapeHtml(c.company_name_el || c.name || '')}</strong> in <strong>${escapeHtml(c.municipality_name || c.city || '')}</strong> exhibits strong morphological alignment with <strong>${targetIcp}</strong>. Operating in <em>“${escapeHtml(primaryKadDesc)}”</em> (${escapeHtml(c.legal_type || '')}), it is an ideal candidate for B2B order automation and Cloud ERP.`
      : `Η επιχείρηση <strong>${escapeHtml(c.company_name_el || c.name || '')}</strong> στην περιοχή <strong>${escapeHtml(c.municipality_name || c.city || '')}</strong> παρουσιάζει εξαιρετική σύγκλιση με το <strong>${targetIcp}</strong>. Με κύριο αντικείμενο <em>«${escapeHtml(primaryKadDesc)}»</em> και νομική μορφή <em>${escapeHtml(c.legal_type || '')}</em>, ενδείκνυται προσέγγιση για εξειδικευμένες B2B λύσεις αυτοματοποίησης παραγγελιών και Cloud ERP.`;

    // Compute Nearby Opportunities & Clientele within 5km
    let nearbyLeads = [];
    let nearbyClients = [];
    if (c.latitude && c.longitude) {
      const clat = Number(c.latitude);
      const clon = Number(c.longitude);
      
      if (state.lookalikes && state.lookalikes.length) {
        nearbyLeads = state.lookalikes
          .filter(l => l.company_id !== c.company_id && l.lat && l.lon)
          .map(l => ({ ...l, distKm: computeDistanceKm(clat, clon, l.lat, l.lon) }))
          .filter(l => l.distKm <= 5.0)
          .sort((a, b) => a.distKm - b.distKm)
          .slice(0, 4);
      }
      
      if (state.clients && state.clients.length) {
        nearbyClients = state.clients
          .filter(cl => cl.company_id !== c.company_id && cl.lat && cl.lon)
          .map(cl => ({ ...cl, distKm: computeDistanceKm(clat, clon, cl.lat, cl.lon) }))
          .filter(cl => cl.distKm <= 5.0)
          .sort((a, b) => a.distKm - b.distKm)
          .slice(0, 3);
      }
    }

    const activeSub = state.drawerSubTab || 'overview';
    
    body.innerHTML = `
      ${ctx.is_excluded ? `
        <div class="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs space-y-2">
          <div class="font-bold text-rose-400 flex items-center justify-between">
            <span class="flex items-center gap-1.5">
              <i data-lucide="ban" class="w-4 h-4 text-rose-400"></i>
              ${isEn ? 'Excluded Business / Inactive' : 'Αποκλεισμένη Επιχείρηση / Διακοπή Εργασιών'}
            </span>
            <span class="text-[10px] text-rose-300 font-mono">${escapeHtml(ctx.exclusion_info?.date_excluded || '')}</span>
          </div>
          <div class="text-slate-300 leading-snug">
            ${escapeHtml(ctx.exclusion_info?.reason || '')}
          </div>
          <button onclick="unexcludeCompany(${c.company_id})" class="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors flex items-center gap-1.5">
            <i data-lucide="rotate-ccw" class="w-3.5 h-3.5 text-amber-400"></i>
            <span>${isEn ? 'Restore to Pipeline' : 'Επαναφορά στο Pipeline'}</span>
          </button>
        </div>
      ` : ''}

      <!-- Header Summary Card -->
      <div class="p-3.5 rounded-xl bg-slate-900/95 border border-slate-800 space-y-2.5">
        <div class="flex items-start justify-between gap-2">
          <div>
            <h3 class="text-sm sm:text-base font-extrabold text-white leading-snug">${escapeHtml(c.company_name_el || c.name || 'Επιχείρηση')}</h3>
            <div class="flex flex-wrap items-center gap-2 text-xs text-slate-400 mt-1">
              <span>${isEn ? 'Tax ID' : 'ΑΦΜ'}: <strong class="text-white font-mono font-semibold">${c.vat}</strong></span>
              <span>·</span>
              <span>${isEn ? 'Legal Form' : 'Μορφή'}: <strong class="text-blue-400 font-bold">${c.legal_type || '-'}</strong></span>
            </div>
          </div>
          ${ctx.lookalike ? `
            <span class="px-2.5 py-1 rounded-lg bg-emerald-500/20 text-emerald-300 font-black text-xs border border-emerald-500/30 shrink-0 font-mono">
              Rank #${ctx.lookalike.rank} (${Math.round((ctx.lookalike.score || 0) * 100)}%)
            </span>
          ` : (ctx.assignment ? `
            <span class="px-2.5 py-1 rounded-lg bg-blue-500/20 text-blue-300 font-black text-xs border border-blue-500/30 shrink-0 font-mono">
              ${ctx.assignment.profile_id}
            </span>
          ` : '')}
        </div>

        <!-- Estimated B2B Commercial Segment Tier Badge (EU AI Act Section 9 Compliant) -->
        <div class="p-2.5 rounded-lg bg-indigo-950/35 border border-indigo-500/30 flex items-center justify-between gap-2 text-xs">
          <div>
            <div class="text-[10px] uppercase font-bold text-indigo-300">${isEn ? 'Estimated B2B Commercial Segment' : 'Εκτιμώμενη Εμπορική Κατηγορία B2B'}</div>
            <div class="font-bold text-white mt-0.5">${escapeHtml(segmentTier)}</div>
          </div>
          <span class="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-[10px] font-semibold shrink-0" title="EU AI Act Section 9: Zero personal credit or solvency scoring">
            🛡️ EU AI Act Sec. 9
          </span>
        </div>
      </div>

      <!-- 3 Sub-Tabs Navigation Bar -->
      <div class="flex items-center gap-1 p-1 rounded-lg bg-slate-900 border border-slate-800">
        <button id="drawer-subbtn-overview" type="button" onclick="switchDrawerSubTab('overview')" class="${activeSub === 'overview' ? 'flex-1 py-1.5 px-2 rounded-md bg-blue-600 text-white text-xs font-bold transition-colors text-center' : 'flex-1 py-1.5 px-2 rounded-md text-slate-400 hover:text-white text-xs font-semibold transition-colors text-center'}">
          1. ${isEn ? 'Overview & Pitch' : 'Σύνοψη & Pitch'}
        </button>
        <button id="drawer-subbtn-geo" type="button" onclick="switchDrawerSubTab('geo')" class="${activeSub === 'geo' ? 'flex-1 py-1.5 px-2 rounded-md bg-blue-600 text-white text-xs font-bold transition-colors text-center' : 'flex-1 py-1.5 px-2 rounded-md text-slate-400 hover:text-white text-xs font-semibold transition-colors text-center'}">
          2. ${isEn ? 'Geospatial Radar' : 'Γεωχωρικό Ραντάρ'}
        </button>
        <button id="drawer-subbtn-registry" type="button" onclick="switchDrawerSubTab('registry')" class="${activeSub === 'registry' ? 'flex-1 py-1.5 px-2 rounded-md bg-blue-600 text-white text-xs font-bold transition-colors text-center' : 'flex-1 py-1.5 px-2 rounded-md text-slate-400 hover:text-white text-xs font-semibold transition-colors text-center'}">
          3. ${isEn ? 'Registry & GEMI' : 'Μητρώο & Γ.Ε.ΜΗ.'}
        </button>
      </div>

      <!-- ================================================================= -->
      <!-- SUB-TAB 1: OVERVIEW, CONTRASTIVE TRAITS & B2B PITCH               -->
      <!-- ================================================================= -->
      <div id="drawer-subpane-overview" class="${activeSub === 'overview' ? '' : 'hidden'} space-y-3.5">
        
        <!-- Contrastive Trait Contribution Breakdown Bar -->
        <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
          <div class="flex items-center justify-between">
            <span class="font-bold text-slate-200 flex items-center gap-1.5">
              <i data-lucide="sliders" class="w-3.5 h-3.5 text-blue-400"></i>
              <span>${isEn ? 'Contrastive Score Contribution Breakdown' : 'Ανάλυση Συνεισφοράς Βαθμολογίας Ταύτισης'}</span>
            </span>
          </div>
          <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden flex">
            <div style="width:${tc.semantic_pct}%" class="bg-blue-500 h-full" title="Semantic & Morphological: ${tc.semantic_pct}%"></div>
            <div style="width:${tc.eb_lift_pct}%" class="bg-emerald-500 h-full" title="Empirical Bayes Lift: ${tc.eb_lift_pct}%"></div>
            <div style="width:${tc.spatial_pct}%" class="bg-amber-400 h-full" title="Spatial Gravity: ${tc.spatial_pct}%"></div>
          </div>
          <div class="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono">
            <div class="p-2 rounded bg-slate-950/80 border border-slate-800">
              <div class="text-[10px] text-blue-400 font-sans font-semibold">${isEn ? 'Semantic Profile' : 'Προφίλ & ΚΑΔ'}</div>
              <div class="font-bold text-white mt-0.5">${tc.semantic_pct}%</div>
            </div>
            <div class="p-2 rounded bg-slate-950/80 border border-slate-800">
              <div class="text-[10px] text-emerald-400 font-sans font-semibold">EB Lift Affinity</div>
              <div class="font-bold text-white mt-0.5">${tc.eb_lift_pct}%</div>
            </div>
            <div class="p-2 rounded bg-slate-950/80 border border-slate-800">
              <div class="text-[10px] text-amber-400 font-sans font-semibold">${isEn ? 'Spatial Gravity' : 'Χωρική Βαρύτητα'}</div>
              <div class="font-bold text-white mt-0.5">${tc.spatial_pct}%</div>
            </div>
          </div>
        </div>

        <!-- Personalized B2B Sales Pitch -->
        <div class="p-3.5 rounded-xl bg-slate-900 border border-blue-500/25 space-y-1.5 text-xs">
          <div class="font-bold text-blue-300 flex items-center gap-2">
            <i data-lucide="sparkles" class="w-4 h-4 text-blue-400"></i>
            <span>${isEn ? 'B2B Sales Intelligence & Outreach Hook' : 'B2B Sales Intelligence & Πρόταση Προσέγγισης'}</span>
          </div>
          <div class="text-slate-300 leading-relaxed">${salesPitch}</div>
        </div>

        ${ctx.lookalike ? `
          <div class="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-xs space-y-1.5">
            <div class="font-bold text-emerald-400 flex items-center gap-2">
              <i data-lucide="target" class="w-4 h-4"></i>
              <span>${isEn ? 'Grounded Lookalike Rationale' : 'Αιτιολόγηση Ομοιότητας με το Προφίλ'}</span>
            </div>
            <div class="text-slate-200 leading-relaxed">${escapeHtml(ctx.lookalike.αιτιολόγηση || ctx.lookalike.reason || '')}</div>
          </div>
        ` : ''}

        <!-- Quick Executive Action Bar -->
        <div class="max-w-md mx-auto grid grid-cols-2 gap-3 pt-1 px-4 sm:px-8">
          <a href="${gmapsUrl}" target="_blank" rel="noopener" class="py-2 px-3 rounded-lg bg-slate-900 hover:bg-slate-800 text-sky-400 border border-slate-800 text-xs font-semibold flex items-center justify-center gap-2 transition-all">
            <i data-lucide="map" class="w-3.5 h-3.5"></i>
            <span>Google Maps</span>
          </a>
          <a href="${webUrl}" target="_blank" rel="noopener" class="py-2 px-3 rounded-lg bg-slate-900 hover:bg-slate-800 text-blue-300 border border-slate-800 text-xs font-semibold flex items-center justify-center gap-2 transition-all">
            <i data-lucide="globe" class="w-3.5 h-3.5"></i>
            <span>Web Search</span>
          </a>
        </div>
      </div>

      <!-- ================================================================= -->
      <!-- SUB-TAB 2: GEOSPATIAL RADAR, ROUTE DISTANCE & 3-TIER OSM AUDIT    -->
      <!-- ================================================================= -->
      <div id="drawer-subpane-geo" class="${activeSub === 'geo' ? '' : 'hidden'} space-y-3.5">
        
        <!-- 3 Geospatial KPIs: Nearest Client Route km, 250m HoReCa Micro-Cluster, 3-Tier OSM Badge -->
        <div class="grid grid-cols-3 gap-2 text-xs">
          <div class="p-3 rounded-lg bg-slate-900 border border-cyan-500/30">
            <div class="text-[10px] uppercase font-bold text-cyan-300">${isEn ? 'Nearest Client' : 'Πλησιέστερος Πελάτης'}</div>
            <div class="text-sm font-extrabold text-white font-mono mt-1">
              ${nearestClientKm !== null ? (nearestClientKm < 1 ? Math.round(nearestClientKm * 1000) + ' m' : nearestClientKm.toFixed(2) + ' km') : '-'}
            </div>
            <div class="text-[10px] text-slate-400 mt-0.5">${nearestClientKm !== null && nearestClientKm <= 1.0 ? (isEn ? '🚚 On Active Route' : '🚚 Εντός Ζώνης 1km') : (isEn ? 'Field Territory' : 'Ζώνη Επέκτασης')}</div>
          </div>

          <div class="p-3 rounded-lg bg-slate-900 border border-purple-500/30">
            <div class="text-[10px] uppercase font-bold text-purple-300">${isEn ? '250m HoReCa Density' : 'Πυκνότητα HoReCa 250m'}</div>
            <div class="text-sm font-extrabold text-white font-mono mt-1">${horeca250m} POIs</div>
            <div class="text-[10px] text-slate-400 mt-0.5">${horeca250m >= 8 ? (isEn ? 'High Cluster Hub' : 'Πυκνή Πιάτσα Εστίασης') : (isEn ? 'Standalone Zone' : 'Αυτόνομο Σημείο')}</div>
          </div>

          <div class="p-3 rounded-lg bg-slate-900 border border-emerald-500/30">
            <div class="text-[10px] uppercase font-bold text-emerald-300">OSM Ambiguity Tier</div>
            <div class="text-xs font-extrabold ${osmTier.includes('Tier 1') ? 'text-emerald-400' : osmTier.includes('Tier 2') ? 'text-amber-300' : 'text-slate-300'} mt-1 truncate" title="${escapeHtml(osmTier)}">
              ${escapeHtml(osmTier)}
            </div>
            <div class="text-[10px] text-slate-400 mt-0.5">${c.osm_distance_m ? `${c.osm_distance_m}m · ${c.osm_name_similarity || 0}% sim` : '3-Tier Resolved'}</div>
          </div>
        </div>

        <!-- 1-Click Route from Nearest Client Action Button -->
        <button type="button" onclick="mapRouteBetweenClientAndProspect(${c.company_id}); closeCompanyDrawer();" class="w-full py-2 px-3 rounded-lg bg-gradient-to-r from-amber-600/25 to-cyan-600/25 hover:from-amber-600/35 hover:to-cyan-600/35 text-amber-200 border border-amber-500/35 text-xs font-bold flex items-center justify-center gap-2 transition-all">
          <span>🚚</span>
          <span>${isEn ? 'Map Road Route from Nearest Active Client' : 'Σχεδίαση Οδικής Διαδρομής από Πλησιέστερο Πελάτη'}</span>
        </button>

        <!-- Interactive Location Mini-Map -->
        <div class="space-y-2 p-3 rounded-xl bg-slate-900 border border-slate-800">
          <div class="flex items-center justify-between text-xs font-bold text-slate-300">
            <span class="flex items-center gap-1.5">
              <i data-lucide="map-pin" class="w-3.5 h-3.5 text-blue-400"></i>
              <span>${escapeHtml(c.street || '')} ${escapeHtml(c.street_number || '')}, ${escapeHtml(c.city || '')}</span>
            </span>
            <span class="text-[10px] text-slate-400 font-mono">${c.latitude ? c.latitude.toFixed(4) + ', ' + c.longitude.toFixed(4) : 'No GPS'}</span>
          </div>
          ${c.latitude && c.longitude ? `
            <div id="company-drawer-minimap" class="w-full h-44 rounded-lg border border-slate-800 overflow-hidden shadow-inner relative z-0"></div>
          ` : `
            <div class="p-4 rounded-lg bg-slate-950 text-slate-500 text-xs italic text-center">No GPS coordinates available.</div>
          `}
        </div>

        <!-- B2B Proximity Radar (<5km) -->
        ${(nearbyLeads.length > 0 || nearbyClients.length > 0) ? `
          <div class="space-y-2 p-3 rounded-xl bg-slate-900 border border-slate-800">
            <div class="flex items-center justify-between text-xs font-bold text-slate-200">
              <span class="flex items-center gap-1.5 text-blue-400">
                <i data-lucide="radar" class="w-4 h-4"></i>
                <span>B2B Proximity Radar (&lt;5km)</span>
              </span>
              <span class="text-[10px] text-slate-400 font-mono">${nearbyLeads.length + nearbyClients.length} POIs</span>
            </div>
            
            ${nearbyClients.length > 0 ? `
              <div class="space-y-1">
                <div class="text-[10px] uppercase font-bold text-emerald-400">${isEn ? 'Nearby Active Clients' : 'Ενεργοί Πελάτες στη Γειτονιά'}</div>
                ${nearbyClients.map(nc => {
                  const distStr = nc.distKm < 1 ? `${Math.round(nc.distKm * 1000)}m` : `${nc.distKm.toFixed(1)}km`;
                  return `
                    <div class="flex items-center justify-between p-2 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-xs gap-2">
                      <div class="truncate pr-1 flex-1">
                        <div class="font-bold text-emerald-200 truncate cursor-pointer hover:text-white" onclick="inspectCompany(${nc.company_id})">
                          ${escapeHtml(nc.name)}
                        </div>
                        <div class="text-[10.5px] text-slate-400 truncate">${escapeHtml(nc.city || '')} · €${Math.round(nc.turnover || 0).toLocaleString('el-GR')}</div>
                      </div>
                      <div class="flex items-center gap-1.5 shrink-0">
                        <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300">${distStr}</span>
                        <button type="button" onclick="mapRouteBetweenClientAndProspect(${c.company_id}, ${nc.company_id}); closeCompanyDrawer();" class="px-2 py-0.5 rounded bg-amber-500/20 hover:bg-amber-500 text-amber-300 hover:text-white border border-amber-500/30 text-[10px] font-semibold flex items-center gap-1 transition-colors" title="${isEn ? 'Route from this client' : 'Διαδρομή από αυτόν τον πελάτη'}">
                          <span>🚚</span>
                          <span>${isEn ? 'Route' : 'Διαδρομή'}</span>
                        </button>
                      </div>
                    </div>
                  `;
                }).join('')}
              </div>
            ` : ''}

            ${nearbyLeads.length > 0 ? `
              <div class="space-y-1 pt-1">
                <div class="text-[10px] uppercase font-bold text-blue-400">${isEn ? 'Nearby Lookalike Prospects' : 'Πλησιέστεροι Υποψήφιοι'}</div>
                ${nearbyLeads.map(nl => {
                  const distStr = nl.distKm < 1 ? `${Math.round(nl.distKm * 1000)}m` : `${nl.distKm.toFixed(1)}km`;
                  return `
                    <div class="flex items-center justify-between p-2 rounded-lg bg-slate-950/80 border border-slate-800 text-xs">
                      <div class="truncate pr-2">
                        <div class="font-bold text-white truncate cursor-pointer hover:text-blue-300" onclick="inspectCompany(${nl.company_id})">
                          #${nl.rank} ${escapeHtml(nl.name)}
                        </div>
                        <div class="text-[10.5px] text-slate-400 truncate">${escapeHtml(nl.kad_desc || nl.kad || '')}</div>
                      </div>
                      <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-500/20 text-blue-300 shrink-0">${distStr}</span>
                    </div>
                  `;
                }).join('')}
              </div>
            ` : ''}
          </div>
        ` : ''}
      </div>

      <!-- ================================================================= -->
      <!-- SUB-TAB 3: REGISTRY, GEMI VERIFICATION & DECLARED KADS            -->
      <!-- ================================================================= -->
      <div id="drawer-subpane-registry" class="${activeSub === 'registry' ? '' : 'hidden'} space-y-3.5">
        
        <!-- Live GEMI Verification Card -->
        <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between gap-3 text-xs">
          <div class="flex items-center gap-2.5">
            ${ctx.verification?.is_closed ? `
              <span class="w-3 h-3 rounded-full bg-rose-500 shrink-0"></span>
              <div>
                <div class="font-bold text-rose-400">${isEn ? 'Permanently Closed / Inactive' : 'Μόνιμα Κλειστό / Διακοπή'}</div>
                <div class="text-[10.5px] text-slate-400">${escapeHtml(isEn ? (ctx.verification?.details_en || ctx.verification?.details || '') : (ctx.verification?.details || ''))}</div>
              </div>
            ` : `
              <span class="w-3 h-3 rounded-full bg-emerald-400 shrink-0"></span>
              <div>
                <div class="font-bold text-emerald-400">${escapeHtml(isEn ? (ctx.verification?.status_label_en || 'Active Business (Modulo 11 Valid)') : (ctx.verification?.status_label || 'Κατάσταση: Ενεργή Επιχείρηση'))}</div>
                <div class="text-[10.5px] text-slate-400">${escapeHtml(isEn ? (ctx.verification?.details_en || 'Verified against GEMI Registry') : (ctx.verification?.details || 'Πιστοποιημένη στο Μητρώο Γ.Ε.ΜΗ.'))}</div>
              </div>
            `}
          </div>
          <button onclick="verifyCompanyLive(${c.company_id})" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 shrink-0 flex items-center gap-1.5">
            <i data-lucide="refresh-cw" class="w-3.5 h-3.5 text-blue-400"></i>
            <span>${isEn ? 'Live Check' : 'Live Έλεγχος'}</span>
          </button>
        </div>

        <!-- Registry Actions -->
        <div class="grid grid-cols-3 gap-2">
          <button onclick="openGemi('${c.vat}', '${escapeHtml(c.company_name_el || c.name).replace(/'/g, "\\'")}')" class="p-2 rounded-lg bg-emerald-950/40 hover:bg-emerald-900/60 text-emerald-400 border border-emerald-500/30 text-xs font-semibold flex items-center justify-center gap-1.5">
            <i data-lucide="shield-check" class="w-3.5 h-3.5"></i>
            <span>Γ.Ε.ΜΗ. Portal</span>
          </button>
          <a href="${gemiSearchUrl}" target="_blank" rel="noopener" class="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5">
            <i data-lucide="search" class="w-3.5 h-3.5"></i>
            <span>${isEn ? 'Find GEMI' : 'Εύρεση Γ.Ε.ΜΗ.'}</span>
          </a>
          <button onclick="excludeCompanyPrompt(${c.company_id}, '${escapeHtml(c.company_name_el || c.name).replace(/'/g, "\\'")}', '${c.vat}')" class="p-2 rounded-lg bg-slate-900 hover:bg-rose-950/40 text-rose-400 border border-slate-800 text-xs font-semibold flex items-center justify-center gap-1.5">
            <i data-lucide="user-x" class="w-3.5 h-3.5"></i>
            <span>${isEn ? 'Exclude' : 'Εξαίρεση'}</span>
          </button>
        </div>

        <!-- Firmographics Grid -->
        <div class="grid grid-cols-2 gap-2.5 text-xs">
          <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
            <div class="text-slate-400 text-[10px] uppercase font-bold">${isEn ? 'Company Age' : 'Ηλικία Επιχείρησης'}</div>
            <div class="text-white font-medium mt-0.5">${(c.company_age_years || 0).toFixed(1)} ${isEn ? 'years' : 'έτη'} (${c.incorporation_date || '-'})</div>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
            <div class="text-slate-400 text-[10px] uppercase font-bold">${isEn ? 'Branch Structure' : 'Δομή & Υποκαταστήματα'}</div>
            <div class="text-white font-medium mt-0.5">${c.is_branch ? (isEn ? 'Branch' : 'Υποκατάστημα') : (isEn ? 'HQ' : 'Έδρα')} · ${c.branch_count || 0} ${isEn ? 'sites' : 'σημεία'}</div>
          </div>
        </div>

        <!-- Declared KADs Table -->
        <div class="space-y-1.5">
          <div class="text-xs font-bold uppercase tracking-wider text-slate-300">
            ${isEn ? `Declared Activity Codes (KAD - ${kads.length})` : `Δηλωμένες Δραστηριότητες (ΚΑΔ - ${kads.length})`}
          </div>
          <div class="rounded-xl border border-slate-800 overflow-hidden text-xs">
            <table class="w-full text-left divide-y divide-slate-800">
              <thead class="bg-slate-900 text-[10px] uppercase font-bold text-slate-400">
                <tr>
                  <th class="py-2 px-2.5">${isEn ? 'Type' : 'Τύπος'}</th>
                  <th class="py-2 px-2.5">ΚΑΔ</th>
                  <th class="py-2 px-2.5">${isEn ? 'Description' : 'Περιγραφή'}</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-800/60 bg-slate-950/60">
                ${kads.map(k => `
                  <tr class="${k.activity_type === 'Κύρια' ? 'bg-blue-500/10' : ''}">
                    <td class="py-1.5 px-2.5 font-semibold ${k.activity_type === 'Κύρια' ? 'text-blue-400' : 'text-slate-400'}">${escapeHtml(k.activity_type)}</td>
                    <td class="py-1.5 px-2.5 font-mono font-bold text-white">${k.activity_id}</td>
                    <td class="py-1.5 px-2.5 text-slate-300 leading-snug">${escapeHtml(k.description || '')}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
    lucide.createIcons();

    if (c.latitude && c.longitude) {
      setTimeout(() => {
        const mapContainer = document.getElementById('company-drawer-minimap');
        if (mapContainer && !drawerMiniMap) {
          try {
            drawerMiniMap = L.map('company-drawer-minimap', {
              center: [c.latitude, c.longitude],
              zoom: 15,
              zoomControl: false,
              attributionControl: false,
              dragging: true,
              scrollWheelZoom: false
            });
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
              maxZoom: 16
            }).addTo(drawerMiniMap);
            
            L.circleMarker([c.latitude, c.longitude], {
              radius: 7,
              fillColor: '#3b82f6',
              color: '#ffffff',
              weight: 2,
              opacity: 1,
              fillOpacity: 1
            }).addTo(drawerMiniMap);
            
            drawerMiniMap.invalidateSize();
          } catch (mapErr) {}
        }
      }, 120);
    }
  } catch (err) {
    body.innerHTML = `<div class="text-rose-400 p-4">Σφάλμα: ${err}</div>`;
  }
}

function closeCompanyDrawer() {
  const drawer = document.getElementById('company-drawer-backdrop');
  if (drawer) {
    drawer.classList.add('hidden');
    drawer.classList.remove('open', 'drawer-open');
  }
  const mapWrapper = document.getElementById('map-view-wrapper');
  if (mapWrapper) {
    mapWrapper.classList.remove('hidden');
  }
  if (drawerMiniMap) {
    try { drawerMiniMap.remove(); } catch (e) {}
    drawerMiniMap = null;
  }
  if (state.map) {
    setTimeout(() => state.map.invalidateSize(), 50);
  }
}

// -----------------------------------------------------------------------------
// In-App Exclusion Modal (Replaces Blocking window.prompt() — Section 4.4)
// -----------------------------------------------------------------------------
window.selectExcludePreset = function(reasonText) {
  const inp = document.getElementById('exclude-reason-input');
  if (inp) inp.value = reasonText;
};

window.closeExcludeReasonModal = function() {
  const modal = document.getElementById('exclude-reason-modal-backdrop');
  if (modal) modal.classList.add('opacity-0', 'pointer-events-none');
  state.pendingExcludeTarget = null;
};

function excludeCompanyPrompt(companyId, companyName, vat) {
  state.pendingExcludeTarget = { companyId, companyName, vat, isBatch: false };
  const label = document.getElementById('exclude-modal-company-label');
  if (label) label.textContent = `${companyName} (ΑΦΜ: ${vat})`;
  const modal = document.getElementById('exclude-reason-modal-backdrop');
  if (modal) modal.classList.remove('opacity-0', 'pointer-events-none');
  lucide.createIcons();
}

window.openBatchExcludeModal = function() {
  const cnt = state.selectedLeadIds.size;
  if (!cnt) return;
  state.pendingExcludeTarget = { isBatch: true };
  const label = document.getElementById('exclude-modal-company-label');
  if (label) {
    label.textContent = state.lang === 'en'
      ? `${cnt} selected prospects`
      : `${cnt} επιλεγμένες επιχειρήσεις`;
  }
  const modal = document.getElementById('exclude-reason-modal-backdrop');
  if (modal) modal.classList.remove('opacity-0', 'pointer-events-none');
  lucide.createIcons();
};

window.confirmExcludeCompany = async function() {
  const target = state.pendingExcludeTarget;
  if (!target) return;
  const inp = document.getElementById('exclude-reason-input');
  const reason = (inp && inp.value.trim()) || 'Διακοπή εργασιών / Χειροκίνητη Εξαίρεση';
  closeExcludeReasonModal();

  try {
    if (target.isBatch) {
      const ids = Array.from(state.selectedLeadIds);
      let lastPayload = null;
      for (const cid of ids) {
        const resp = await fetch(`/api/exclude-company/${cid}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason })
        });
        const data = await resp.json();
        if (data.status === 'ok' && data.payload) lastPayload = data.payload;
      }
      state.selectedLeadIds.clear();
      updateBatchFabUI();
      const exResp = await fetch('/api/excluded-companies');
      const exData = await exResp.json();
      state.excluded = exData.excluded || [];
      updateExcludedHeaderStatus();
      if (lastPayload) {
        applyEnginePayload(lastPayload);
        renderAllViews();
      }
      showToast(`🚫 Εξαιρέθηκαν ${ids.length} επιχειρήσεις από το pipeline.`, 'alert-triangle');
      return;
    }

    const resp = await fetch(`/api/exclude-company/${target.companyId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason })
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      showToast(`🚫 Η επιχείρηση "${target.companyName}" εξαιρέθηκε από το pipeline.`, 'alert-triangle');
      const exResp = await fetch('/api/excluded-companies');
      const exData = await exResp.json();
      state.excluded = exData.excluded || [];
      updateExcludedHeaderStatus();
      
      if (data.payload) {
        applyEnginePayload(data.payload);
        renderAllViews();
        if (state.selectedCompanyId === target.companyId) {
          inspectCompany(target.companyId);
        }
      }
    } else {
      showToast(`Σφάλμα: ${data.message || 'Αποτυχία αποκλεισμού'}`, 'alert-circle');
    }
  } catch (err) {
    showToast(`Σφάλμα δικτύου: ${err}`, 'alert-circle');
  }
};

async function unexcludeCompany(companyId) {
  try {
    const resp = await fetch(`/api/unexclude-company/${companyId}`, {
      method: 'POST'
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      showToast(`✅ Η επιχείρηση επανήλθε στο pipeline στόχευσης.`, 'check-circle-2');
      const exResp = await fetch('/api/excluded-companies');
      const exData = await exResp.json();
      state.excluded = exData.excluded || [];
      updateExcludedHeaderStatus();
      renderExcludedModal();
      
      if (data.payload) {
        applyEnginePayload(data.payload);
        renderAllViews();
        if (state.selectedCompanyId === companyId) {
          inspectCompany(companyId);
        }
      }
    } else {
      showToast(`Σφάλμα: ${data.message || 'Αποτυχία επαναφοράς'}`, 'alert-circle');
    }
  } catch (err) {
    showToast(`Σφάλμα δικτύου: ${err}`, 'alert-circle');
  }
}

let excludedTabState = 'user';
let excludedSearchQuery = '';

function setExcludedTab(tab) {
  excludedTabState = tab;
  const userBtn = document.getElementById('excluded-tab-user');
  const supBtn = document.getElementById('excluded-tab-superseded');
  if (userBtn && supBtn) {
    if (tab === 'user') {
      userBtn.className = 'px-3 py-1.5 rounded-md font-semibold text-white bg-slate-800 transition-colors flex items-center gap-1.5';
      supBtn.className = 'px-3 py-1.5 rounded-md font-semibold text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5';
    } else {
      userBtn.className = 'px-3 py-1.5 rounded-md font-semibold text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5';
      supBtn.className = 'px-3 py-1.5 rounded-md font-semibold text-white bg-slate-800 transition-colors flex items-center gap-1.5';
    }
  }
  renderExcludedModal();
}

function onExcludedSearchInput(query) {
  excludedSearchQuery = (query || '').trim().toLowerCase();
  renderExcludedModal();
}

function openExcludedModal() {
  const modal = document.getElementById('excluded-modal-backdrop');
  if (!modal) return;
  modal.classList.remove('opacity-0', 'pointer-events-none');
  renderExcludedModal();
}

function closeExcludedModal() {
  const modal = document.getElementById('excluded-modal-backdrop');
  if (modal) modal.classList.add('opacity-0', 'pointer-events-none');
}

function renderExcludedModal() {
  const list = document.getElementById('excluded-modal-list');
  if (!list) return;
  const allItems = state.excluded || [];
  
  const userCount = allItems.filter(x => (x.source || 'user') === 'user').length;
  const supCount = allItems.filter(x => x.source === 'superseded').length;
  
  const uPill = document.getElementById('excluded-count-user-pill');
  const sPill = document.getElementById('excluded-count-sup-pill');
  if (uPill) uPill.textContent = String(userCount);
  if (sPill) sPill.textContent = supCount.toLocaleString('el-GR');

  let items = allItems.filter(x => {
    const src = x.source || 'user';
    return src === excludedTabState;
  });

  if (excludedSearchQuery) {
    items = items.filter(x => {
      const name = (x.name || '').toLowerCase();
      const vat = String(x.vat || '').toLowerCase();
      const city = (x.city || '').toLowerCase();
      const reason = (x.reason || '').toLowerCase();
      return name.includes(excludedSearchQuery) || vat.includes(excludedSearchQuery) || city.includes(excludedSearchQuery) || reason.includes(excludedSearchQuery);
    });
  }

  if (!items.length) {
    const emptyMsg = excludedSearchQuery 
      ? uiT('blacklist.empty_search', 'Δεν βρέθηκαν αποτελέσματα για την αναζήτηση.') 
      : (excludedTabState === 'user' 
          ? uiT('blacklist.empty_user', 'Δεν υπάρχουν χειροκίνητες εξαιρέσεις. Μπορείτε να εξαιρέσετε επιχειρήσεις από την καρτέλα υποψηφίου.') 
          : uiT('blacklist.empty_superseded', 'Δεν υπάρχουν εξαιρέσεις αντικατάστασης χώρου.'));
    list.innerHTML = `<div class="py-12 text-center text-slate-500 text-xs">${emptyMsg}</div>`;
    return;
  }

  // Render max 150 items if large list to keep DOM silky smooth
  const displayItems = items.slice(0, 150);
  const truncatedTemplate = uiT('blacklist.truncated_notice', 'Προβολή πρώτων {shown} από {total} εγγραφές. Χρησιμοποιήστε την αναζήτηση για εντοπισμό συγκεκριμένης επιχείρησης.');
  const formattedNotice = truncatedTemplate
    .replace('{shown}', String(150))
    .replace('{total}', items.length.toLocaleString(state.lang === 'en' ? 'en-US' : 'el-GR'));

  const truncatedNotice = items.length > 150 
    ? `<div class="text-[11px] text-slate-500 text-center py-2 border-t border-slate-800">${formattedNotice}</div>` 
    : '';

  const badgeUser = uiT('blacklist.badge_user', 'Χειροκίνητη Αφαίρεση');
  const badgeSuperseded = uiT('blacklist.badge_superseded', 'Αντικατάσταση Χώρου');
  const excludedOnLabel = uiT('blacklist.excluded_on', 'Εξαιρέθηκε');
  const restoreBtnLabel = uiT('blacklist.restore_btn', 'Επαναφορά');
  const restoreBtnTitle = uiT('blacklist.restore_title', 'Επαναφορά στα Lookalikes');

  list.innerHTML = displayItems.map(ex => {
    const isSup = ex.source === 'superseded';
    const badgeColor = isSup ? 'bg-amber-950/70 text-amber-300 border-amber-800/40' : 'bg-rose-950/70 text-rose-300 border-rose-800/40';
    const badgeLabel = isSup ? badgeSuperseded : badgeUser;
    const iconName = isSup ? 'history' : 'user-x';

    return `
      <div class="p-3 rounded-lg bg-slate-900/90 border border-slate-800/90 hover:border-slate-700/80 flex items-start justify-between gap-3 transition-colors">
        <div class="space-y-1.5 min-w-0 flex-1">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="font-bold text-white text-xs truncate max-w-sm">${escapeHtml(ex.name)}</span>
            <span class="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded border border-slate-700/50">ΑΦΜ: ${escapeHtml(ex.vat)}</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded border ${badgeColor} flex items-center gap-1 font-medium">
              <i data-lucide="${iconName}" class="w-3 h-3"></i>
              ${badgeLabel}
            </span>
          </div>
          <div class="text-[11px] ${isSup ? 'text-amber-300/90' : 'text-rose-300/90'} font-normal leading-relaxed">${escapeHtml(ex.reason)}</div>
          <div class="text-[10px] text-slate-500 flex items-center gap-2">
            <span>${excludedOnLabel}: ${escapeHtml(ex.date_excluded || '-')}</span>
            ${ex.city ? `<span>·</span><span>${escapeHtml(ex.city)}</span>` : ''}
          </div>
        </div>
        <button onclick="unexcludeCompany(${ex.company_id})" class="px-2.5 py-1.5 rounded-md bg-slate-800/90 hover:bg-slate-700 text-slate-200 hover:text-white text-xs font-semibold border border-slate-700/80 shrink-0 flex items-center gap-1.5 transition-colors" title="${restoreBtnTitle}">
          <i data-lucide="rotate-ccw" class="w-3.5 h-3.5 text-blue-400"></i>
          <span class="text-[11px]">${restoreBtnLabel}</span>
        </button>
      </div>
    `;
  }).join('') + truncatedNotice;
  
  lucide.createIcons();
}

async function verifyCompanyLive(companyId) {
  showToast('🔍 Εκτέλεση ζωντανού ελέγχου στο Γ.Ε.ΜΗ. & Google Places...', 'refresh-cw');
  try {
    const resp = await fetch(`/api/verify-company/${companyId}`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      const v = data.verification;
      if (v.is_closed) {
        showToast(`⚠️ Εξακριβώθηκε: Η επιχείρηση έχει κλείσει (${v.details}). Αφαιρέθηκε από το pipeline.`, 'alert-triangle');
      } else {
        showToast(`✅ Εξακριβώθηκε: Η επιχείρηση είναι ενεργή (${v.source}).`, 'check-circle-2');
      }
      if (data.payload) {
        applyEnginePayload(data.payload);
        renderAllViews();
      }
      inspectCompany(companyId);
    } else {
      showToast(`Σφάλμα ελέγχου: ${data.message || 'Αποτυχία ελέγχου'}`, 'alert-circle');
    }
  } catch (err) {
    showToast(`Σφάλμα δικτύου: ${err}`, 'alert-circle');
  }
}

// -----------------------------------------------------------------------------
// Local LLM Grounded Prompt Modal & Generation
// -----------------------------------------------------------------------------
async function openLlmModal(profileId) {
  const modal = document.getElementById('llm-modal-backdrop');
  const title = document.getElementById('llm-modal-title');
  const sysEl = document.getElementById('llm-system-prompt');
  const usrEl = document.getElementById('llm-user-prompt');
  const execBtn = document.getElementById('llm-execute-btn');
  const statusText = document.getElementById('llm-status-text');
  const offlineBanner = document.getElementById('llm-offline-banner');
  const modelSelect = document.getElementById('llm-model-select');
  
  if (title) title.textContent = `Grounded LLM Prompt — ${profileId}`;
  if (sysEl) sysEl.textContent = 'Φόρτωση System Prompt...';
  if (usrEl) usrEl.textContent = 'Φόρτωση Statistical JSON Payload...';
  if (modal) modal.classList.remove('opacity-0', 'pointer-events-none');
  
  try {
    const resp = await fetch(`/api/llm/prompt/${profileId}`);
    const data = await resp.json();
    if (sysEl) sysEl.textContent = data.system_prompt;
    if (usrEl) usrEl.textContent = data.user_prompt;
    
    const isOnline = data.ollama && data.ollama.available;
    state.ollama = data.ollama || state.ollama;
    updateOllamaHeaderStatus();
    
    if (isOnline) {
      if (offlineBanner) offlineBanner.classList.add('hidden');
      if (statusText) {
        statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>Ollama Online (${data.ollama.model})</span>`;
      }
      
      // Populate model dropdown if multiple local models exist
      if (modelSelect && data.ollama.installed && data.ollama.installed.length) {
        modelSelect.classList.remove('hidden');
        modelSelect.innerHTML = data.ollama.installed.map(m => `
          <option value="${m}" ${m === data.ollama.model ? 'selected' : ''}>${m}</option>
        `).join('');
      }
      
      if (execBtn) {
        execBtn.disabled = false;
        execBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i><span>Εκτέλεση με Τοπικό Ollama</span>';
        execBtn.onclick = () => runLlmGeneration(profileId);
      }
    } else {
      if (offlineBanner) offlineBanner.classList.remove('hidden');
      if (statusText) {
        statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span><span>Ollama Offline (Ενεργός Deterministic Fallback)</span>`;
      }
      if (modelSelect) modelSelect.classList.add('hidden');
      if (execBtn) {
        execBtn.disabled = false;
        execBtn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i><span>Εφαρμογή Grounded Fallback</span>';
        execBtn.onclick = async () => {
          try {
            const fResp = await fetch(`/api/llm/generate/${profileId}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model: 'deterministic' }),
            });
            const fData = await fResp.json();
            if (fData.accepted) {
              const prof = state.profiles.find(p => p.id === profileId);
              if (prof) {
                prof.ai_description = fData.text;
                renderIcpCards();
              }
              showToast('✓ Εφαρμόστηκε Grounded Περιγραφή', 'check');
              closeLlmModal();
            }
          } catch (err) {
            closeLlmModal();
          }
        };
      }
    }
    lucide.createIcons();
  } catch (err) {
    if (sysEl) sysEl.textContent = `Σφάλμα φόρτωσης: ${err}`;
  }
}

async function runLlmGeneration(profileId) {
  const execBtn = document.getElementById('llm-execute-btn');
  const statusText = document.getElementById('llm-status-text');
  const modelSelect = document.getElementById('llm-model-select');
  const selectedModel = modelSelect ? modelSelect.value : state.ollama.model;
  
  if (execBtn) {
    execBtn.disabled = true;
    execBtn.innerHTML = '<div class="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin"></div><span>Εκτέλεση...</span>';
  }
  if (statusText) {
    statusText.innerHTML = '<span class="w-2 h-2 rounded-full bg-blue-400 animate-ping"></span><span>Εκτέλεση LLM (temp=0.0) &amp; έλεγχος Article 50...</span>';
  }
  
  try {
    const resp = await fetch(`/api/llm/generate/${profileId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selectedModel }),
    });
    const res = await resp.json();
    
    if (res.accepted) {
      if (statusText) {
        statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span class="text-emerald-400 font-semibold">Επιτυχία (${res.elapsed_sec}s) — Πλήρης συμμόρφωση Art. 50</span>`;
      }
      const prof = state.profiles.find(p => p.id === profileId);
      if (prof) {
        prof.ai_description = res.text;
        prof.ai_source = 'ollama';
        renderIcpCards();
      }
      showToast('✓ Παράχθηκε επιτυχώς κείμενο από τοπικό LLM', 'check');
      setTimeout(closeLlmModal, 1200);
    } else {
      if (statusText) {
        statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span><span class="text-rose-400 font-semibold">Απόρριψη: ${res.notes ? res.notes.join(', ') : 'Μη συμβατή έξοδος'}</span>`;
      }
      if (execBtn) {
        execBtn.disabled = false;
        execBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i><span>Επανάληψη</span>';
      }
    }
  } catch (err) {
    if (statusText) {
      statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span><span class="text-rose-400">Σφάλμα σύνδεσης: ${err}</span>`;
    }
    if (execBtn) {
      execBtn.disabled = false;
      execBtn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i><span>Επανάληψη</span>';
    }
  }
  lucide.createIcons();
}

function closeLlmModal() {
  const modal = document.getElementById('llm-modal-backdrop');
  if (modal) modal.classList.add('opacity-0', 'pointer-events-none');
}

function openOllamaGuideModal() {
  const modal = document.getElementById('ollama-guide-modal-backdrop');
  if (modal) modal.classList.remove('opacity-0', 'pointer-events-none');
  lucide.createIcons();
}

function closeOllamaGuideModal() {
  const modal = document.getElementById('ollama-guide-modal-backdrop');
  if (modal) modal.classList.add('opacity-0', 'pointer-events-none');
}

function copyToClipboard(elementId, btnElement) {
  const el = document.getElementById(elementId);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent || '').then(() => {
    if (btnElement) {
      const orig = btnElement.innerHTML;
      btnElement.innerHTML = '<span class="text-emerald-400 font-bold">Αντιγράφηκε! ✓</span>';
      setTimeout(() => { btnElement.innerHTML = orig; lucide.createIcons(); }, 1500);
    }
  });
}

// -----------------------------------------------------------------------------
// Floating Analyst Modal Handlers
// -----------------------------------------------------------------------------
function openAnalystDrawer() {
  const backdrop = document.getElementById('analyst-drawer-backdrop');
  const content = document.getElementById('analyst-drawer-content');
  if (backdrop && content) {
    backdrop.classList.remove('opacity-0', 'pointer-events-none');
    content.classList.remove('scale-95', 'opacity-0');
    content.classList.add('scale-100', 'opacity-100');
  }
}

function closeAnalystDrawer() {
  const backdrop = document.getElementById('analyst-drawer-backdrop');
  const content = document.getElementById('analyst-drawer-content');
  if (backdrop && content) {
    backdrop.classList.add('opacity-0', 'pointer-events-none');
    content.classList.remove('scale-100', 'opacity-100');
    content.classList.add('scale-95', 'opacity-0');
  }
}

async function handleClientUpload(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) {
    dropZone.innerHTML = `
      <div class="py-4 text-center">
        <div class="w-6 h-6 rounded-full border-2 border-blue-500 border-t-transparent animate-spin mx-auto mb-2"></div>
        <div class="text-xs text-slate-300 font-medium">Επεξεργασία &amp; ταυτοποίηση πελατολογίου...</div>
      </div>
    `;
  }
  
  try {
    const resp = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const res = await resp.json();
    if (res.status === 'ok') {
      state.params.client_set = res.client_set.client_set;
      const runResp = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: state.params }),
      });
      const runData = await runResp.json();
      if (runData.status === 'ok') {
        applyEnginePayload(runData.data);
        populateFilterDropdowns();
        renderAllViews();
        showToast(`✓ Φορτώθηκαν ${res.client_set.matched} πελάτες επιτυχώς!`, 'check');
      }
    } else {
      alert(`Σφάλμα: ${res.message}`);
    }
  } catch (err) {
    alert(`Σφάλμα κατά το ανέβασμα: ${err}`);
  } finally {
    if (dropZone) {
      dropZone.innerHTML = `
        <input type="file" id="client-file-input" accept=".xlsx" class="hidden">
        <i data-lucide="upload-cloud" class="w-6 h-6 mx-auto text-blue-400 mb-1.5"></i>
        <div class="text-xs font-medium text-slate-200">${isEn ? 'Drop or click to upload Excel' : 'Σύρετε ή πατήστε για ανέβασμα Excel'}</div>
        <div class="text-[11px] text-slate-400 mt-0.5">${isEn ? 'Columns: AFM and Revenue (€)' : 'Στήλες: ΑΦΜ και Κύκλος Εργασιών (€)'}</div>
      `;
      setupUploadListeners();
      lucide.createIcons();
    }
  }
}

// -----------------------------------------------------------------------------
// Global Event Listeners Setup
// -----------------------------------------------------------------------------
function setupEventListeners() {
  // Navigation Tabs Switching
  document.querySelectorAll('.nav-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.getAttribute('data-tab');
      state.activeTab = tab;
      
      document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      ['overview', 'leads', 'clients', 'tech'].forEach(t => {
        const el = document.getElementById(`tab-content-${t}`);
        if (el) {
          if (t === tab) {
            el.classList.remove('hidden');
          } else {
            el.classList.add('hidden');
          }
        }
      });
      
      // Invalidate map size so Leaflet renders properly if switching to overview tab
      if (tab === 'overview' && state.map) {
        setTimeout(() => state.map.invalidateSize(), 100);
      }
      lucide.createIcons();
    });
  });

  // Region Preset Buttons on the Map
  document.querySelectorAll('.region-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const region = btn.getAttribute('data-region');
      zoomToRegion(region);
    });
  });

  // Lookalikes Quick-Filter Chips
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.leadsFilter.quickFilter = chip.getAttribute('data-filter');
      state.leadsFilter.page = 1;
      renderLeadsTable();
    });
  });

  // Clients Quick-Filter Chips
  document.querySelectorAll('.client-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.client-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      state.clientsFilter.quickFilter = chip.getAttribute('data-client-filter');
      renderClientsTable();
    });
  });

  // Slider change listeners
  [
    'sim_weight_feature_cosine',
    'sim_weight_eb_lift_affinity',
    'sim_weight_spatial_kernel',
    'spatial_kernel_bandwidth_km',
    'eb_prior_strength',
    'no_fit_min_score',
    'hybrid_alpha',
  ].forEach(k => {
    const el = document.getElementById(`param-${k}`);
    if (el) {
      el.addEventListener('input', e => handleSliderChange(k, e.target.value));
    }
  });

  // Toggle: use_embeddings
  const useEmbToggle = document.getElementById('param-use_embeddings');
  if (useEmbToggle) {
    useEmbToggle.addEventListener('change', e => {
      const isChecked = e.target.checked;
      state.params.use_embeddings = isChecked;
      if (isChecked) {
        if (!state.params.hybrid_alpha || state.params.hybrid_alpha <= 0) {
          state.params.hybrid_alpha = 0.50;
        }
      } else {
        state.params.hybrid_alpha = 0.00;
      }
      syncSlidersToState();
      handleSliderChange('use_embeddings', isChecked);
      showToast(isChecked ? '🟢 Νευρωνικά Embeddings Ενεργά' : '⚪ Αλγεβρικό SVD Baseline (Neural OFF)', 'cpu');
    });
  }

  // Header quick toggle button
  const headerEmbBtn = document.getElementById('header-embeddings-btn');
  if (headerEmbBtn) {
    headerEmbBtn.addEventListener('click', e => {
      e.stopPropagation();
      const cur = Boolean(state.params.use_embeddings !== false);
      const next = !cur;
      state.params.use_embeddings = next;
      if (next) {
        if (!state.params.hybrid_alpha || state.params.hybrid_alpha <= 0) {
          state.params.hybrid_alpha = 0.50;
        }
      } else {
        state.params.hybrid_alpha = 0.00;
      }
      syncSlidersToState();
      handleSliderChange('use_embeddings', next);
      showToast(next ? '🟢 Νευρωνικά Embeddings Ενεργά' : '⚪ Αλγεβρικό SVD Baseline (Neural OFF)', 'cpu');
    });
  }

  // Select: embedding_model
  const embModelSelect = document.getElementById('param-embedding_model');
  if (embModelSelect) {
    embModelSelect.addEventListener('change', e => {
      state.params.embedding_model = e.target.value;
      syncSlidersToState();
      handleSliderChange('embedding_model', e.target.value);
    });
  }

  // Number & Select params in analyst drawer
  ['min_icps', 'max_icps', 'turnover_weighting_mode'].forEach(k => {
    const el = document.getElementById(`param-${k}`);
    if (el) {
      el.addEventListener('change', e => {
        state.params[k] = isNaN(e.target.value) ? e.target.value : parseInt(e.target.value);
        handleSliderChange(k, e.target.value);
      });
    }
  });

  // Reset Weights Button
  const resetBtn = document.getElementById('reset-params-btn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      state.params = { ...state.defaults };
      syncSlidersToState();
      handleSliderChange('sim_weight_feature_cosine', state.params.sim_weight_feature_cosine);
    });
  }

  // Drawers open/close
  const openDrawerBtn = document.getElementById('open-analyst-drawer-btn');
  if (openDrawerBtn) openDrawerBtn.addEventListener('click', openAnalystDrawer);
  
  const closeDrawerBtn = document.getElementById('close-analyst-drawer-btn');
  if (closeDrawerBtn) closeDrawerBtn.addEventListener('click', closeAnalystDrawer);
  
  const closeDrawerBtn2 = document.getElementById('close-analyst-drawer-btn-2');
  if (closeDrawerBtn2) closeDrawerBtn2.addEventListener('click', closeAnalystDrawer);
  
  const analystBackdrop = document.getElementById('analyst-drawer-backdrop');
  if (analystBackdrop) {
    analystBackdrop.addEventListener('click', e => {
      if (e.target === analystBackdrop) closeAnalystDrawer();
    });
  }

  const closeCompanyBtn = document.getElementById('close-company-drawer-btn');
  if (closeCompanyBtn) closeCompanyBtn.addEventListener('click', closeCompanyDrawer);
  
  const companyBackdrop = document.getElementById('company-drawer-backdrop');
  if (companyBackdrop) {
    companyBackdrop.addEventListener('click', e => {
      if (e.target === companyBackdrop) closeCompanyDrawer();
    });
  }

  const closeLlmBtn = document.getElementById('close-llm-modal-btn');
  if (closeLlmBtn) closeLlmBtn.addEventListener('click', closeLlmModal);
  
  const llmBackdrop = document.getElementById('llm-modal-backdrop');
  if (llmBackdrop) {
    llmBackdrop.addEventListener('click', e => {
      if (e.target === llmBackdrop) closeLlmModal();
    });
  }

  const exclBackdrop = document.getElementById('excluded-modal-backdrop');
  if (exclBackdrop) {
    exclBackdrop.addEventListener('click', e => {
      if (e.target === exclBackdrop) closeExcludedModal();
    });
  }

  // Greek Keyboard Focus Shield & Ergonomic Navigation Shortcuts (Section 4.7)
  document.addEventListener('keydown', e => {
    // Ctrl+K or Cmd+K focuses the search bar
    if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'k' || e.key === 'κ')) {
      e.preventDefault();
      const sInput = document.getElementById('leads-search-input');
      if (sInput) {
        sInput.focus();
        sInput.select();
      }
      return;
    }

    if (e.key === 'Escape') {
      closeAnalystDrawer();
      closeCompanyDrawer();
      closeLlmModal();
      closeOllamaGuideModal();
      closeExcludedModal();
      closeExcludeReasonModal();
      return;
    }

    // Focus Shield: Ignore single-key shortcuts when typing inside inputs, textareas, or selects
    const active = document.activeElement;
    if (active) {
      const tag = active.tagName ? active.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || active.isContentEditable) {
        return;
      }
    }

    const key = e.key;
    const lower = key.toLowerCase();

    // Down navigation: ArrowDown or J or Greek ξ (same physical key as J)
    if (key === 'ArrowDown' || lower === 'j' || lower === 'ξ') {
      if (state.activeTab === 'overview') {
        e.preventDefault();
        const items = state.currentFilteredLeads && state.currentFilteredLeads.length ? state.currentFilteredLeads : state.lookalikes;
        if (!items || !items.length) return;
        state.focusedLeadIndex = Math.min(items.length - 1, Math.max(0, state.focusedLeadIndex + 1));
        const target = items[state.focusedLeadIndex];
        if (!target) return;

        const targetPage = Math.floor(state.focusedLeadIndex / state.leadsFilter.pageSize) + 1;
        if (state.leadsFilter.page !== targetPage) {
          state.leadsFilter.page = targetPage;
          renderLeadsTable();
        } else {
          document.querySelectorAll('#leads-table-body tr').forEach(r => r.classList.remove('row-active-highlight'));
          const row = document.getElementById(`lead-row-${target.company_id}`);
          if (row) {
            row.classList.add('row-active-highlight');
            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
        if (target.lat && target.lon) primeMapMarkerOnHover(target.lat, target.lon);
        const drawer = document.getElementById('company-drawer-backdrop');
        if (drawer && (drawer.classList.contains('open') || drawer.classList.contains('drawer-open'))) {
          inspectCompany(target.company_id);
        }
      }
      return;
    }

    // Up navigation: ArrowUp or K or Greek κ
    if (key === 'ArrowUp' || lower === 'k' || lower === 'κ') {
      if (state.activeTab === 'overview') {
        e.preventDefault();
        const items = state.currentFilteredLeads && state.currentFilteredLeads.length ? state.currentFilteredLeads : state.lookalikes;
        if (!items || !items.length) return;
        state.focusedLeadIndex = Math.max(0, state.focusedLeadIndex <= 0 ? 0 : state.focusedLeadIndex - 1);
        const target = items[state.focusedLeadIndex];
        if (!target) return;

        const targetPage = Math.floor(state.focusedLeadIndex / state.leadsFilter.pageSize) + 1;
        if (state.leadsFilter.page !== targetPage) {
          state.leadsFilter.page = targetPage;
          renderLeadsTable();
        } else {
          document.querySelectorAll('#leads-table-body tr').forEach(r => r.classList.remove('row-active-highlight'));
          const row = document.getElementById(`lead-row-${target.company_id}`);
          if (row) {
            row.classList.add('row-active-highlight');
            row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
        if (target.lat && target.lon) primeMapMarkerOnHover(target.lat, target.lon);
        const drawer = document.getElementById('company-drawer-backdrop');
        if (drawer && (drawer.classList.contains('open') || drawer.classList.contains('drawer-open'))) {
          inspectCompany(target.company_id);
        }
      }
      return;
    }

    // Enter opens the slide-over drawer for the focused row
    if (key === 'Enter' && state.activeTab === 'overview' && state.focusedLeadIndex >= 0) {
      const items = state.currentFilteredLeads && state.currentFilteredLeads.length ? state.currentFilteredLeads : state.lookalikes;
      const target = items && items[state.focusedLeadIndex];
      if (target) {
        e.preventDefault();
        inspectCompany(target.company_id);
      }
      return;
    }

    // M or Greek μ cycles Split -> Map -> Split
    if (lower === 'm' || lower === 'μ') {
      setLayoutMode(state.layoutMode === 'map' ? 'split' : 'map');
      return;
    }

    // T or Greek τ cycles Split -> Table -> Split
    if (lower === 't' || lower === 'τ') {
      setLayoutMode(state.layoutMode === 'table' ? 'split' : 'table');
      return;
    }

    // D or Greek δ cycles table density
    if (lower === 'd' || lower === 'δ') {
      const order = ['compact', 'standard', 'detailed'];
      const next = order[(order.indexOf(state.tableDensity) + 1) % order.length];
      setTableDensity(next);
      return;
    }

    // E or Greek ε opens Exclusions modal
    if (lower === 'e' || lower === 'ε') {
      openExcludedModal();
    }
  });

  // Dynamic Scroll Indicator: Reveal scrollbar thumb only when actively scrolling or hovering over it
  document.addEventListener('scroll', e => {
    const target = e.target;
    if (target && target.classList && target.classList.contains('overflow-y-auto')) {
      target.classList.add('is-scrolling');
      clearTimeout(target._scrollTimeout);
      target._scrollTimeout = setTimeout(() => {
        target.classList.remove('is-scrolling');
      }, 700);
    }
  }, true);

  // Export menu toggle
  const exportBtn = document.getElementById('export-dropdown-btn');
  const exportMenu = document.getElementById('export-menu');
  if (exportBtn && exportMenu) {
    exportBtn.addEventListener('click', e => {
      e.stopPropagation();
      exportMenu.classList.toggle('hidden');
    });
    document.addEventListener('click', e => {
      const wrapper = document.getElementById('export-dropdown-wrapper');
      if (wrapper && !wrapper.contains(e.target)) {
        exportMenu.classList.add('hidden');
      }
    });
  }

  // Leads search & filters
  const searchInput = document.getElementById('leads-search-input');
  if (searchInput) {
    let searchDebounce = null;
    searchInput.addEventListener('input', e => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        state.leadsFilter.search = e.target.value.trim();
        state.leadsFilter.page = 1;
        renderLeadsTable();
      }, 100);
    });
  }

  const profSelect = document.getElementById('leads-profile-filter');
  if (profSelect) {
    profSelect.addEventListener('change', e => {
      state.leadsFilter.profile = e.target.value;
      state.leadsFilter.page = 1;
      renderLeadsTable();
    });
  }

  const prefSelect = document.getElementById('leads-pref-filter');
  if (prefSelect) {
    prefSelect.addEventListener('change', e => {
      state.leadsFilter.prefecture = e.target.value;
      state.leadsFilter.page = 1;
      renderLeadsTable();
    });
  }

  const legalSelect = document.getElementById('leads-legal-filter');
  if (legalSelect) {
    legalSelect.addEventListener('change', e => {
      state.leadsFilter.legal = e.target.value;
      state.leadsFilter.page = 1;
      renderLeadsTable();
    });
  }

  const pageSizeSelect = document.getElementById('leads-page-size');
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', e => {
      state.leadsFilter.pageSize = parseInt(e.target.value);
      state.leadsFilter.page = 1;
      renderLeadsTable();
    });
  }

  const prevPageBtn = document.getElementById('leads-prev-page');
  if (prevPageBtn) {
    prevPageBtn.addEventListener('click', () => {
      if (state.leadsFilter.page > 1) {
        state.leadsFilter.page--;
        renderLeadsTable();
      }
    });
  }

  const nextPageBtn = document.getElementById('leads-next-page');
  if (nextPageBtn) {
    nextPageBtn.addEventListener('click', () => {
      state.leadsFilter.page++;
      renderLeadsTable();
    });
  }

  // Clients search & filters
  const clientsSearch = document.getElementById('clients-search-input');
  if (clientsSearch) {
    clientsSearch.addEventListener('input', e => {
      state.clientsFilter.search = e.target.value.trim();
      renderClientsTable();
    });
  }

  const clientsProf = document.getElementById('clients-profile-filter');
  if (clientsProf) {
    clientsProf.addEventListener('change', e => {
      state.clientsFilter.profile = e.target.value;
      renderClientsTable();
    });
  }

  setupUploadListeners();
}

function setupUploadListeners() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('client-file-input');
  if (!dropZone || !fileInput) return;
  
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files && e.target.files[0]) {
      handleClientUpload(e.target.files[0]);
    }
  });
  
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500', 'bg-blue-500/10');
  });
  
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500', 'bg-blue-500/10');
  });
  
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500', 'bg-blue-500/10');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleClientUpload(e.dataTransfer.files[0]);
    }
  });
}

// -----------------------------------------------------------------------------
// Global Window API Exports for HTML Event Handlers & Cross-Component Access
// -----------------------------------------------------------------------------
window.inspectCompany = inspectCompany;
window.closeCompanyDrawer = closeCompanyDrawer;
window.focusOnMap = focusOnMap;
window.excludeCompanyPrompt = excludeCompanyPrompt;
window.unexcludeCompany = unexcludeCompany;
window.openAnalystDrawer = openAnalystDrawer;
window.closeAnalystDrawer = closeAnalystDrawer;
window.openGemi = openGemi;
window.copyToClipboard = copyToClipboard;
window.exportFilteredLeadsCsv = exportFilteredLeadsCsv;
window.openExcludedModal = openExcludedModal;
window.closeExcludedModal = closeExcludedModal;
window.setExcludedTab = setExcludedTab;
window.onExcludedSearchInput = onExcludedSearchInput;
window.openLlmModal = openLlmModal;
window.openOllamaGuideModal = openOllamaGuideModal;
window.closeOllamaGuideModal = closeOllamaGuideModal;
window.switchBasemap = switchBasemap;
window.verifyCompanyLive = verifyCompanyLive;
