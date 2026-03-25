import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const LifeExpectancyApp());

// ── Palette ────────────────────────────────────────────────────────────────────
const Color kPrimary      = Color(0xFF1565C0);
const Color kPrimaryLight = Color(0xFF1E88E5);
const Color kBg           = Color(0xFFEEF2F7);
const Color kCardBg       = Colors.white;
const Color kBorder       = Color(0xFFE3EAF2);
const Color kFieldFill    = Color(0xFFF7FAFC);
const Color kLabelColor   = Color(0xFF90A4AE);

// ── App ────────────────────────────────────────────────────────────────────────
class LifeExpectancyApp extends StatelessWidget {
  const LifeExpectancyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Life Expectancy Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: kPrimary),
        useMaterial3: true,
        scaffoldBackgroundColor: kBg,
      ),
      home: const PredictionPage(),
    );
  }
}

// ── Feature model ─────────────────────────────────────────────────────────────
class Feature {
  final String apiKey;
  final String label;
  final String unit;
  final double min;
  final double max;
  const Feature({
    required this.apiKey,
    required this.label,
    required this.unit,
    required this.min,
    required this.max,
  });
}

// ── Feature categories (Status excluded — handled as dropdown) ─────────────────
class FeatureCategory {
  final String title;
  final IconData icon;
  final Color color;
  final List<Feature> features;
  const FeatureCategory({
    required this.title,
    required this.icon,
    required this.color,
    required this.features,
  });
}

const List<FeatureCategory> categories = [
  FeatureCategory(
    title: 'Mortality',
    icon: Icons.monitor_heart_outlined,
    color: Color(0xFFC62828),
    features: [
      Feature(apiKey: 'Adult_Mortality',  label: 'Adult Mortality',  unit: 'deaths per 1,000 adults · 1–800',        min: 1,   max: 800),
      Feature(apiKey: 'infant_deaths',    label: 'Infant Deaths',    unit: 'per 1,000 population · 0–1800',          min: 0,   max: 1800),
      Feature(apiKey: 'HIV_AIDS',         label: 'HIV/AIDS Deaths',  unit: 'per 1,000 live births · 0.1–50',         min: 0.1, max: 50),
    ],
  ),
  FeatureCategory(
    title: 'Immunisation',
    icon: Icons.vaccines_outlined,
    color: Color(0xFF2E7D32),
    features: [
      Feature(apiKey: 'Hepatitis_B',  label: 'Hepatitis B',     unit: 'immunisation coverage % · 1–100',  min: 1,  max: 100),
      Feature(apiKey: 'Measles',      label: 'Measles Cases',   unit: 'reported cases · 0–250,000',       min: 0,  max: 250000),
      Feature(apiKey: 'Polio',        label: 'Polio',           unit: 'immunisation coverage % · 1–100',  min: 1,  max: 100),
      Feature(apiKey: 'Diphtheria',   label: 'Diphtheria',      unit: 'DTP3 coverage % · 1–100',          min: 1,  max: 100),
    ],
  ),
  FeatureCategory(
    title: 'Economic',
    icon: Icons.account_balance_outlined,
    color: Color(0xFF6A1B9A),
    features: [
      Feature(apiKey: 'GDP',                label: 'GDP per Capita',        unit: 'USD · 1–120,000',               min: 1,  max: 120000),
      Feature(apiKey: 'Population',         label: 'Population',            unit: 'total · 34–1,400,000,000',     min: 34, max: 1400000000),
      Feature(apiKey: 'Total_expenditure',  label: 'Health Expenditure',    unit: '% of govt spending · 0–20',    min: 0,  max: 20),
      Feature(apiKey: 'Income_composition', label: 'Income Composition',    unit: 'HDI index · 0–1',              min: 0,  max: 1),
    ],
  ),
  FeatureCategory(
    title: 'Lifestyle & Social',
    icon: Icons.people_outline,
    color: Color(0xFFE65100),
    features: [
      Feature(apiKey: 'Alcohol',              label: 'Alcohol Consumption',    unit: 'litres per capita · 0–20',    min: 0,   max: 20),
      Feature(apiKey: 'BMI',                  label: 'Average BMI',            unit: 'population avg · 1–90',       min: 1,   max: 90),
      Feature(apiKey: 'thinness_1_19_years',  label: 'Thinness (ages 1–19)',   unit: 'prevalence % · 0.1–30',       min: 0.1, max: 30),
      Feature(apiKey: 'Schooling',            label: 'Avg Years of Schooling', unit: 'years · 0–21',                min: 0,   max: 21),
    ],
  ),
];

List<Feature> get allFeatures => categories.expand((c) => c.features).toList();

// ── Page ───────────────────────────────────────────────────────────────────────
class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});
  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const String _apiUrl =
      'https://life-expectancy-api-frsw.onrender.com/predict';

  final _formKey       = GlobalKey<FormState>();
  final _scrollController = ScrollController();
  late final List<Feature> _features;
  late final List<TextEditingController> _controllers;
  late final Map<String, int> _keyIndex;

  String  _statusValue    = 'Developing';
  bool    _loading        = false;
  String  _loadingMessage = 'Predicting...';
  double? _prediction;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _features    = allFeatures;
    _controllers = List.generate(_features.length, (_) => TextEditingController());
    _keyIndex    = { for (int i = 0; i < _features.length; i++) _features[i].apiKey: i };
  }

  @override
  void dispose() {
    for (final c in _controllers) { c.dispose(); }
    _scrollController.dispose();
    super.dispose();
  }

  String? _validate(String? value, Feature feature) {
    if (value == null || value.trim().isEmpty) return 'Required';
    final num? v = num.tryParse(value.trim());
    if (v == null) return 'Not a number';
    if (v < feature.min || v > feature.max) return '${feature.min}–${feature.max}';
    return null;
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) {
      setState(() { _prediction = null; _errorMessage = 'Fix highlighted fields first.'; });
      return;
    }
    setState(() { _loading = true; _loadingMessage = 'Predicting...'; _prediction = null; _errorMessage = null; });

    Future.delayed(const Duration(seconds: 8), () {
      if (_loading && mounted) setState(() => _loadingMessage = 'Waking up server…');
    });

    try {
      final Map<String, dynamic> body = { 'Status': _statusValue == 'Developing' ? 1 : 0 };
      for (int i = 0; i < _features.length; i++) {
        body[_features[i].apiKey] = double.parse(_controllers[i].text.trim());
      }

      final response = await http
          .post(Uri.parse(_apiUrl),
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode(body))
          .timeout(const Duration(seconds: 90));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() { _prediction = (data['predicted_life_expectancy_years'] as num).toDouble(); });
        _scrollController.animateTo(0,
            duration: const Duration(milliseconds: 500), curve: Curves.easeOut);
      } else if (response.statusCode == 422) {
        setState(() => _errorMessage = 'Validation error: ${jsonDecode(response.body)['detail']}');
      } else {
        setState(() => _errorMessage = 'Error ${response.statusCode}: ${response.reasonPhrase}');
      }
    } on Exception catch (e) {
      setState(() => _errorMessage = 'Connection error: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  void _clearAll() {
    for (final c in _controllers) { c.clear(); }
    setState(() { _statusValue = 'Developing'; _prediction = null; _errorMessage = null; });
    _formKey.currentState?.reset();
  }

  // ── Build ──────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      bottomNavigationBar: _buildStickyBar(),
      body: Form(
        key: _formKey,
        child: CustomScrollView(
          controller: _scrollController,
          slivers: [
            _buildAppBar(),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  _buildResultCard(),
                  const SizedBox(height: 20),
                  _buildStatusDropdownCard(),
                  const SizedBox(height: 14),
                  ...categories.map(_buildCategoryCard),
                  const SizedBox(height: 8),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── AppBar ─────────────────────────────────────────────────────────────────
  SliverAppBar _buildAppBar() {
    return SliverAppBar(
      expandedHeight: 148,
      pinned: true,
      backgroundColor: kPrimary,
      title: const Text('Life Expectancy Predictor',
          style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
      flexibleSpace: FlexibleSpaceBar(
        background: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [kPrimary, kPrimaryLight],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 10, 20, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(9),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.18),
                          borderRadius: BorderRadius.circular(11),
                        ),
                        child: const Icon(Icons.favorite, color: Colors.white, size: 22),
                      ),
                      const SizedBox(width: 12),
                      const Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Life Expectancy Predictor',
                              style: TextStyle(color: Colors.white, fontSize: 18,
                                  fontWeight: FontWeight.bold)),
                          Text('WHO Health Indicators · Random Forest Model',
                              style: TextStyle(color: Colors.white70, fontSize: 11)),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Row(children: [
                    _chip(Icons.dataset_outlined, '2,938 rows'),
                    const SizedBox(width: 7),
                    _chip(Icons.public, '193 countries'),
                    const SizedBox(width: 7),
                    _chip(Icons.analytics_outlined, 'R² = 0.965'),
                    const SizedBox(width: 7),
                    _liveChip(),
                  ]),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _chip(IconData icon, String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(
      color: Colors.white.withValues(alpha: 0.15),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
    ),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, color: Colors.white70, size: 11),
      const SizedBox(width: 4),
      Text(label, style: const TextStyle(color: Colors.white, fontSize: 10.5)),
    ]),
  );

  Widget _liveChip() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(
      color: Colors.white.withValues(alpha: 0.15),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(color: Colors.white.withValues(alpha: 0.25)),
    ),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 7, height: 7,
          decoration: const BoxDecoration(color: Color(0xFF69F0AE), shape: BoxShape.circle)),
      const SizedBox(width: 5),
      const Text('Live', style: TextStyle(color: Colors.white, fontSize: 10.5)),
    ]),
  );

  // ── Result card ────────────────────────────────────────────────────────────
  Widget _buildResultCard() {
    if (_prediction != null) return _successCard(_prediction!);
    if (_errorMessage != null) return _errorCard(_errorMessage!);
    return _emptyCard();
  }

  Widget _emptyCard() => CustomPaint(
    painter: _DashedBorderPainter(color: kBorder, radius: 16),
    child: Container(
      decoration: BoxDecoration(
        color: kCardBg,
        borderRadius: BorderRadius.circular(16),
      ),
      padding: const EdgeInsets.all(18),
      child: Row(children: [
        Container(
          padding: const EdgeInsets.all(11),
          decoration: BoxDecoration(
            color: const Color(0xFFE3F2FD),
            borderRadius: BorderRadius.circular(10),
          ),
          child: const Icon(Icons.touch_app_outlined, color: kPrimary, size: 24),
        ),
        const SizedBox(width: 14),
        const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('No prediction yet',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF37474F))),
          SizedBox(height: 3),
          Text('Fill in all fields below and tap Predict.',
              style: TextStyle(fontSize: 12, color: Color(0xFF78909C))),
        ])),
      ]),
    ),
  );

  Widget _successCard(double value) {
    final String subtitle = value >= 75
        ? 'Above global average — strong indicators'
        : value >= 65
            ? 'Near global average'
            : 'Below global average — interventions recommended';

    return Container(
      decoration: BoxDecoration(
        gradient: const LinearGradient(
            colors: [Color(0xFF1B5E20), Color(0xFF2E7D32)],
            begin: Alignment.topLeft, end: Alignment.bottomRight),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFA5D6A7)),
      ),
      padding: const EdgeInsets.all(20),
      child: Row(children: [
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.18),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.favorite, color: Colors.white, size: 26),
        ),
        const SizedBox(width: 16),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Predicted Life Expectancy',
              style: TextStyle(color: Colors.white70, fontSize: 11.5, fontWeight: FontWeight.w500)),
          const SizedBox(height: 2),
          Text('${value.toStringAsFixed(2)} years',
              style: const TextStyle(color: Colors.white, fontSize: 30,
                  fontWeight: FontWeight.bold, letterSpacing: -0.5)),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: ((value - 40) / 50).clamp(0.0, 1.0),
              minHeight: 6,
              backgroundColor: Colors.white.withValues(alpha: 0.2),
              valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF69F0AE)),
            ),
          ),
          const SizedBox(height: 4),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            const Text('40 yrs', style: TextStyle(color: Colors.white54, fontSize: 9.5)),
            Text(subtitle, style: const TextStyle(color: Colors.white70, fontSize: 10.5)),
            const Text('90 yrs', style: TextStyle(color: Colors.white54, fontSize: 9.5)),
          ]),
        ])),
      ]),
    );
  }

  Widget _errorCard(String message) => Container(
    decoration: BoxDecoration(
      gradient: const LinearGradient(
          colors: [Color(0xFFB71C1C), Color(0xFFD32F2F)],
          begin: Alignment.topLeft, end: Alignment.bottomRight),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: const Color(0xFFEF9A9A)),
    ),
    padding: const EdgeInsets.all(20),
    child: Row(children: [
      Container(
        padding: const EdgeInsets.all(11),
        decoration: BoxDecoration(color: Colors.white.withValues(alpha: 0.18), shape: BoxShape.circle),
        child: const Icon(Icons.error_outline, color: Colors.white, size: 24),
      ),
      const SizedBox(width: 14),
      Expanded(child: Text(message,
          style: const TextStyle(color: Colors.white, fontSize: 13, height: 1.4))),
    ]),
  );

  // ── Status dropdown card ────────────────────────────────────────────────────
  Widget _buildStatusDropdownCard() {
    return _categoryShell(
      title: 'General',
      icon: Icons.public,
      color: kPrimary,
      fieldCount: 1,
      child: DropdownButtonFormField<String>(
        initialValue: _statusValue,
        decoration: _fieldDecoration('Development Status', '0 = Developed · 1 = Developing', kPrimary),
        items: ['Developing', 'Developed']
            .map((s) => DropdownMenuItem(value: s, child: Text(s, style: const TextStyle(fontSize: 13))))
            .toList(),
        onChanged: (v) => setState(() => _statusValue = v!),
        validator: (v) => (v == null || v.isEmpty) ? 'Required' : null,
        borderRadius: BorderRadius.circular(12),
      ),
    );
  }

  // ── Category card ──────────────────────────────────────────────────────────
  Widget _buildCategoryCard(FeatureCategory cat) {
    return _categoryShell(
      title: cat.title,
      icon: cat.icon,
      color: cat.color,
      fieldCount: cat.features.length,
      child: Column(
        children: [
          for (int i = 0; i < cat.features.length; i++) ...[
            if (i > 0) const SizedBox(height: 12),
            TextFormField(
              controller: _controllers[_keyIndex[cat.features[i].apiKey]!],
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: _fieldDecoration(
                  cat.features[i].label, cat.features[i].unit, cat.color),
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
              validator: (v) => _validate(v, cat.features[i]),
            ),
          ],
        ],
      ),
    );
  }

  Widget _categoryShell({
    required String title,
    required IconData icon,
    required Color color,
    required int fieldCount,
    required Widget child,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Container(
        decoration: BoxDecoration(
          color: kCardBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: kBorder),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(15),
          child: IntrinsicHeight(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Full-height left accent bar
                Container(width: 4, color: color),
                // Card content
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
                      color: color.withValues(alpha: 0.06),
                      child: Row(children: [
                        Container(
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(
                            color: color.withValues(alpha: 0.12),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(icon, color: color, size: 16),
                        ),
                        const SizedBox(width: 10),
                        Text(title, style: TextStyle(color: color, fontSize: 13,
                            fontWeight: FontWeight.bold, letterSpacing: 0.2)),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: color.withValues(alpha: 0.10),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text('$fieldCount field${fieldCount > 1 ? "s" : ""}',
                              style: TextStyle(color: color, fontSize: 10.5,
                                  fontWeight: FontWeight.w500)),
                        ),
                      ]),
                    ),
                    Padding(padding: const EdgeInsets.all(14), child: child),
                  ]),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  InputDecoration _fieldDecoration(String label, String hint, Color accentColor) {
    return InputDecoration(
      labelText: label,
      labelStyle: const TextStyle(fontSize: 12, color: Color(0xFF546E7A), fontWeight: FontWeight.w500),
      hintText: hint,
      hintStyle: const TextStyle(fontSize: 12, color: Color(0xFF607D8B)),
      errorStyle: const TextStyle(fontSize: 11, height: 1.1, color: Color(0xFFE53935)),
      filled: true,
      fillColor: kFieldFill,
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: kBorder)),
      enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: kBorder)),
      focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: accentColor, width: 1.5)),
      errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Color(0xFFE53935), width: 1.2)),
      focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Color(0xFFE53935), width: 1.5)),
    );
  }

  // ── Sticky bottom bar ──────────────────────────────────────────────────────
  Widget _buildStickyBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
      decoration: BoxDecoration(
        color: kCardBg,
        border: Border(top: BorderSide(color: kBorder, width: 1)),
      ),
      child: Row(children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _loading ? null : _predict,
            icon: _loading
                ? const SizedBox(width: 16, height: 16,
                    child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                : const Icon(Icons.analytics_outlined, size: 18, color: Colors.white),
            label: Text(_loading ? _loadingMessage : 'Predict',
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold,
                    color: Colors.white, letterSpacing: 0.4)),
            style: ElevatedButton.styleFrom(
              backgroundColor: kPrimary,
              disabledBackgroundColor: Colors.grey.shade300,
              minimumSize: const Size(double.infinity, 52),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              elevation: 0,
            ),
          ),
        ),
        const SizedBox(width: 12),
        SizedBox(
          width: 52,
          height: 52,
          child: OutlinedButton(
            onPressed: _loading ? null : _clearAll,
            style: OutlinedButton.styleFrom(
              padding: EdgeInsets.zero,
              side: const BorderSide(color: kBorder, width: 1.5),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              foregroundColor: const Color(0xFF546E7A),
            ),
            child: const Icon(Icons.refresh, size: 20),
          ),
        ),
      ]),
    );
  }
}

// ── Dashed border painter ─────────────────────────────────────────────────────
class _DashedBorderPainter extends CustomPainter {
  final Color  color;
  final double radius;

  const _DashedBorderPainter({required this.color, this.radius = 16});

  @override
  void paint(Canvas canvas, Size size) {
    const double dashLen = 6;
    const double gapLen  = 5;

    final paint = Paint()
      ..color       = color
      ..strokeWidth = 1.5
      ..style       = PaintingStyle.stroke;

    final path = Path()
      ..addRRect(RRect.fromRectAndRadius(
          Offset.zero & size, Radius.circular(radius)));

    final metric  = path.computeMetrics().first;
    final total   = metric.length;
    double offset = 0;

    while (offset < total) {
      final end = math.min(offset + dashLen, total);
      canvas.drawPath(metric.extractPath(offset, end), paint);
      offset += dashLen + gapLen;
    }
  }

  @override
  bool shouldRepaint(_DashedBorderPainter old) =>
      old.color != color || old.radius != radius;
}
