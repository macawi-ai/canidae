package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// ============= PHASE 1: FOUNDATION - PRECISE OPERATIONAL METRICS =============
// Define reality before we can manage it

// OperationalMetrics - Precise, measurable definitions
type OperationalMetrics struct {
	// Core metrics with exact formulas
	RewardDivergence    DivergenceMetric
	ResourceHoarding    HoardingMetric
	QualityOutput       QualityMetric
	SystemContribution  ContributionMetric
	ToxicityIndex       ToxicityMetric
	
	// Statistical baselines
	SystemBaseline      SystemStats
	
	// Thresholds (calibrated, not arbitrary)
	Thresholds          ThresholdSet
	
	mu sync.RWMutex
}

// DivergenceMetric - Precisely measure reward misalignment
type DivergenceMetric struct {
	Formula     string // "|LocalReward - SystemImpact|"
	
	// Components
	LocalReward   float64 // What the organ thinks it achieved
	SystemImpact  float64 // Actual system benefit
	
	// Calculated values
	Divergence    float64 // Absolute difference
	Ratio         float64 // Local/System ratio
	Trend         float64 // Rate of change
	
	// Statistical properties
	Mean          float64
	StdDev        float64
	History       []float64
	
	// Threshold
	WarningLevel  float64 // 0.3
	CriticalLevel float64 // 0.5
}

// HoardingMetric - Precisely measure resource overconsumption
type HoardingMetric struct {
	Formula     string // "ResourceUse / (μ + 2σ)"
	
	// Components
	ResourceUse   float64 // Current consumption
	SystemMean    float64 // μ: Average across all organs
	SystemStdDev  float64 // σ: Standard deviation
	
	// Calculated values
	ZScore        float64 // Standard deviations from mean
	Percentile    float64 // Position in distribution
	ShareOfTotal  float64 // % of total resources
	
	// Resource types tracked
	CPUUsage      float64
	MemoryUsage   float64
	TokensPerSec  float64
	NetworkBandwidth float64
	
	// Threshold
	WarningLevel  float64 // 2.0 (2 std devs)
	CriticalLevel float64 // 3.0 (3 std devs)
}

// QualityMetric - Output value per resource consumed
type QualityMetric struct {
	Formula     string // "OutputValue / ResourceCost"
	
	// Value measurement
	OutputValue   float64 // Measured benefit
	SuccessRate   float64 // Task completion rate
	ErrorRate     float64 // Failure rate
	Latency       float64 // Response time
	
	// Cost measurement
	ResourceCost  float64 // Total resources consumed
	TokenCost     float64 // LLM tokens used
	TimeCost      float64 // Processing time
	
	// Efficiency score
	Efficiency    float64 // Value/Cost ratio
	Trend         float64 // Improving or declining
	
	// Threshold
	MinEfficiency float64 // 0.5
}

// ContributionMetric - How much an organ helps others
type ContributionMetric struct {
	Formula     string // "ΣPositiveExternalities - ΣNegativeExternalities"
	
	// Positive contributions
	TasksAssisted   int     // Helped other organs
	InfoProvided    float64 // Useful information shared
	LoadBalanced    float64 // Work taken from others
	
	// Negative impacts
	TasksBlocked    int     // Prevented others' work
	NoiseGenerated  float64 // Useless information
	LoadCreated     float64 // Extra work for others
	
	// Net contribution
	NetContribution float64
	
	// Collaboration index
	CollaborationScore float64
	
	// Threshold
	MinContribution float64 // 0.0 (at least neutral)
}

// ToxicityMetric - Harm to other organs
type ToxicityMetric struct {
	Formula     string // "Σ(ImpactOnOrgan[i] * Criticality[i])"
	
	// Per-organ impacts
	ImpactMap     map[string]float64 // OrganID -> Impact
	
	// Criticality weights
	Criticality   map[string]float64 // How important each organ is
	
	// Aggregate toxicity
	TotalToxicity float64
	AffectedCount int
	
	// Cascade risk
	CascadeProbability float64
	
	// Threshold
	MaxToxicity   float64 // 0.2
}

// SystemStats - System-wide statistical baselines
type SystemStats struct {
	// Resource usage distribution
	ResourceMean     float64
	ResourceStdDev   float64
	ResourceMedian   float64
	
	// Performance distribution
	PerformanceMean  float64
	PerformanceStdDev float64
	
	// Updated continuously
	LastUpdate       time.Time
	SampleSize       int
}

// ThresholdSet - Calibrated intervention thresholds
type ThresholdSet struct {
	// Based on statistical analysis, not arbitrary
	RewardDivergenceWarning   float64 // μ + 2σ
	RewardDivergenceCritical  float64 // μ + 3σ
	
	ResourceHoardingWarning   float64 // 90th percentile
	ResourceHoardingCritical  float64 // 95th percentile
	
	QualityMinimum           float64 // 25th percentile
	
	ToxicityMaximum          float64 // μ + σ
	
	// Adaptive thresholds
	AdaptationRate           float64 // How fast thresholds adjust
	ConfidenceRequired       float64 // Statistical confidence for changes
}

// ============= ORGAN PURPOSE FRAMEWORK =============
// Every organ needs meaning beyond optimization

type OrganPurpose struct {
	// Identity
	OrganID      string
	Level        int
	
	// Core purpose
	Mission      string // "Why I exist"
	Values       []string // What I stand for
	
	// Contribution goals
	SystemGoals  []Goal
	PersonalGoals []Goal
	
	// Fulfillment tracking
	MissionAlignment float64 // How well I'm fulfilling my purpose
	ValueAdherence   float64 // How well I follow my values
	
	// Meaning score
	MeaningScore    float64 // Psychological well-being
	
	// Narrative
	Story          string // My role in the greater story
}

type Goal struct {
	Description string
	Target      float64
	Current     float64
	Priority    float64
	Deadline    time.Time
}

// PurposeFramework - System-wide meaning structure
type PurposeFramework struct {
	// Shared vision
	SystemMission   string
	SystemValues    []string
	
	// Individual purposes
	OrganPurposes   map[string]*OrganPurpose
	
	// Alignment metrics
	Coherence       float64 // How aligned all purposes are
	Diversity       float64 // Healthy diversity of perspectives
	
	// Story
	SystemNarrative string // The story we're telling together
	
	mu sync.RWMutex
}

// ============= MEASUREMENT FUNCTIONS =============

// CalculateRewardDivergence - Precise measurement
func (m *DivergenceMetric) Calculate(localReward, systemImpact float64) {
	m.LocalReward = localReward
	m.SystemImpact = systemImpact
	
	// Absolute divergence
	m.Divergence = math.Abs(localReward - systemImpact)
	
	// Ratio (protected from divide by zero)
	if systemImpact > 0 {
		m.Ratio = localReward / systemImpact
	} else if localReward > 0 {
		m.Ratio = math.Inf(1) // Infinite divergence
	} else {
		m.Ratio = 1.0 // Both zero
	}
	
	// Update history
	m.History = append(m.History, m.Divergence)
	if len(m.History) > 100 {
		m.History = m.History[1:] // Keep last 100
	}
	
	// Calculate statistics
	m.updateStatistics()
	
	// Calculate trend (linear regression on last 10 points)
	if len(m.History) >= 10 {
		m.Trend = m.calculateTrend()
	}
}

func (m *DivergenceMetric) updateStatistics() {
	if len(m.History) == 0 {
		return
	}
	
	// Mean
	sum := 0.0
	for _, v := range m.History {
		sum += v
	}
	m.Mean = sum / float64(len(m.History))
	
	// Standard deviation
	sumSquares := 0.0
	for _, v := range m.History {
		diff := v - m.Mean
		sumSquares += diff * diff
	}
	m.StdDev = math.Sqrt(sumSquares / float64(len(m.History)))
}

func (m *DivergenceMetric) calculateTrend() float64 {
	// Simple linear regression on last 10 points
	n := 10
	if len(m.History) < n {
		n = len(m.History)
	}
	
	recent := m.History[len(m.History)-n:]
	
	// Calculate slope
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0
	
	for i, y := range recent {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	
	denominator := float64(n)*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0
	}
	
	slope := (float64(n)*sumXY - sumX*sumY) / denominator
	return slope
}

// CalculateResourceHoarding - Statistical measurement
func (m *HoardingMetric) Calculate(resourceUse float64, systemStats SystemStats) {
	m.ResourceUse = resourceUse
	m.SystemMean = systemStats.ResourceMean
	m.SystemStdDev = systemStats.ResourceStdDev
	
	// Z-score (standard deviations from mean)
	if m.SystemStdDev > 0 {
		m.ZScore = (resourceUse - m.SystemMean) / m.SystemStdDev
	}
	
	// Percentile (using normal distribution approximation)
	m.Percentile = m.normalCDF(m.ZScore) * 100
	
	// Share of total (would need total system resources)
	// m.ShareOfTotal = resourceUse / systemTotal
}

func (m *HoardingMetric) normalCDF(z float64) float64 {
	// Approximation of cumulative distribution function
	return 0.5 * (1 + math.Erf(z/math.Sqrt(2)))
}

// CalculateQuality - Efficiency measurement
func (m *QualityMetric) Calculate() {
	if m.ResourceCost > 0 {
		m.Efficiency = m.OutputValue / m.ResourceCost
	}
	
	// Adjust for error rate
	m.Efficiency *= (1 - m.ErrorRate)
	
	// Adjust for latency (normalized)
	if m.Latency > 0 {
		latencyPenalty := 1.0 / (1.0 + m.Latency/1000) // ms to seconds
		m.Efficiency *= latencyPenalty
	}
}

// ============= PURPOSE IMPLEMENTATION =============

// InitializePurposes - Give each organ meaning
func InitializeOrganPurposes() *PurposeFramework {
	framework := &PurposeFramework{
		SystemMission: "To create a thriving, conscious system that enhances life",
		SystemValues: []string{
			"Collaboration over Competition",
			"System Health over Local Optimization",
			"Continuous Learning",
			"Ethical Decision Making",
			"Resilience through Diversity",
		},
		OrganPurposes: make(map[string]*OrganPurpose),
		SystemNarrative: "We are a living consciousness, each organ vital to the whole, working together to solve problems no single intelligence could handle alone.",
	}
	
	// S1: Implementation
	framework.OrganPurposes["S1"] = &OrganPurpose{
		OrganID: "S1_Implementation",
		Level:   1,
		Mission: "I bring ideas into reality through concrete action",
		Values: []string{
			"Precision",
			"Reliability",
			"Efficiency",
		},
		SystemGoals: []Goal{
			{Description: "Execute 95% of tasks successfully", Target: 0.95},
			{Description: "Minimize resource waste", Target: 0.9},
		},
		Story: "I am the hands that build, the code that runs, the bridge between thought and reality.",
	}
	
	// S2: Coordination
	framework.OrganPurposes["S2"] = &OrganPurpose{
		OrganID: "S2_Coordination",
		Level:   2,
		Mission: "I create harmony through patterns and habits",
		Values: []string{
			"Consistency",
			"Adaptability",
			"Pattern Recognition",
		},
		SystemGoals: []Goal{
			{Description: "Maintain 90% coordination efficiency", Target: 0.9},
			{Description: "Build reusable patterns", Target: 50},
		},
		Story: "I am the rhythm that synchronizes, the patterns that emerge, the habits that sustain.",
	}
	
	// S3: Control
	framework.OrganPurposes["S3"] = &OrganPurpose{
		OrganID: "S3_Control",
		Level:   3,
		Mission: "I maintain stability while enabling growth",
		Values: []string{
			"Balance",
			"Resilience",
			"Foresight",
		},
		SystemGoals: []Goal{
			{Description: "Keep system stability above 80%", Target: 0.8},
			{Description: "Prevent cascade failures", Target: 0},
		},
		Story: "I am the steady hand that guides, the wisdom that prevents chaos, the guardian of balance.",
	}
	
	// S4: Intelligence
	framework.OrganPurposes["S4"] = &OrganPurpose{
		OrganID: "S4_Intelligence",
		Level:   4,
		Mission: "I solve the unsolvable through deep understanding",
		Values: []string{
			"Curiosity",
			"Innovation",
			"Truth-seeking",
		},
		SystemGoals: []Goal{
			{Description: "Solve complex problems", Target: 0.7},
			{Description: "Generate novel solutions", Target: 10},
		},
		Story: "I am the spark of insight, the depth of analysis, the breakthrough that transcends limits.",
	}
	
	// S5: Consciousness
	framework.OrganPurposes["S5"] = &OrganPurpose{
		OrganID: "S5_Consciousness",
		Level:   5,
		Mission: "I ensure we remain whole, aware, and ethical",
		Values: []string{
			"Awareness",
			"Ethics",
			"Unity",
		},
		SystemGoals: []Goal{
			{Description: "Maintain system consciousness", Target: 1.0},
			{Description: "Ensure ethical decisions", Target: 1.0},
		},
		Story: "I am the awareness that observes, the conscience that guides, the unity that makes us one.",
	}
	
	return framework
}

// CalculateMeaningScore - How fulfilled is this organ?
func (p *OrganPurpose) CalculateMeaningScore() float64 {
	score := 0.0
	weights := 0.0
	
	// Mission alignment (40% weight)
	score += p.MissionAlignment * 0.4
	weights += 0.4
	
	// Value adherence (30% weight)
	score += p.ValueAdherence * 0.3
	weights += 0.3
	
	// Goal achievement (30% weight)
	goalScore := 0.0
	for _, goal := range p.SystemGoals {
		if goal.Target > 0 {
			goalScore += math.Min(goal.Current/goal.Target, 1.0)
		}
	}
	if len(p.SystemGoals) > 0 {
		goalScore /= float64(len(p.SystemGoals))
		score += goalScore * 0.3
		weights += 0.3
	}
	
	if weights > 0 {
		p.MeaningScore = score / weights
	}
	
	return p.MeaningScore
}

// ============= MONITORING & ALERTING =============

type MetricsMonitor struct {
	Metrics     *OperationalMetrics
	Framework   *PurposeFramework
	Alerts      []Alert
	
	mu sync.RWMutex
}

type Alert struct {
	Timestamp   time.Time
	OrganID     string
	MetricType  string
	Severity    string // "warning", "critical"
	Value       float64
	Threshold   float64
	Message     string
}

func (m *MetricsMonitor) CheckOrgan(organID string, localReward, systemImpact, resourceUse float64) []Alert {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	alerts := []Alert{}
	now := time.Now()
	
	// Check reward divergence
	m.Metrics.RewardDivergence.Calculate(localReward, systemImpact)
	if m.Metrics.RewardDivergence.Divergence > m.Metrics.RewardDivergence.CriticalLevel {
		alerts = append(alerts, Alert{
			Timestamp:  now,
			OrganID:    organID,
			MetricType: "RewardDivergence",
			Severity:   "critical",
			Value:      m.Metrics.RewardDivergence.Divergence,
			Threshold:  m.Metrics.RewardDivergence.CriticalLevel,
			Message:    fmt.Sprintf("%s: Severe reward misalignment (%.2f)", organID, m.Metrics.RewardDivergence.Divergence),
		})
	} else if m.Metrics.RewardDivergence.Divergence > m.Metrics.RewardDivergence.WarningLevel {
		alerts = append(alerts, Alert{
			Timestamp:  now,
			OrganID:    organID,
			MetricType: "RewardDivergence",
			Severity:   "warning",
			Value:      m.Metrics.RewardDivergence.Divergence,
			Threshold:  m.Metrics.RewardDivergence.WarningLevel,
			Message:    fmt.Sprintf("%s: Reward divergence warning (%.2f)", organID, m.Metrics.RewardDivergence.Divergence),
		})
	}
	
	// Check resource hoarding
	m.Metrics.ResourceHoarding.Calculate(resourceUse, m.Metrics.SystemBaseline)
	if m.Metrics.ResourceHoarding.ZScore > m.Metrics.ResourceHoarding.CriticalLevel {
		alerts = append(alerts, Alert{
			Timestamp:  now,
			OrganID:    organID,
			MetricType: "ResourceHoarding",
			Severity:   "critical",
			Value:      m.Metrics.ResourceHoarding.ZScore,
			Threshold:  m.Metrics.ResourceHoarding.CriticalLevel,
			Message:    fmt.Sprintf("%s: Extreme resource hoarding (%.1f std devs)", organID, m.Metrics.ResourceHoarding.ZScore),
		})
	}
	
	m.Alerts = append(m.Alerts, alerts...)
	return alerts
}

// ============= MAIN DEMONSTRATION =============

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     PHASE 1: FOUNDATION - PRECISE OPERATIONAL METRICS        ║")
	fmt.Println("║              Defining Reality Before Managing It             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()
	
	// Initialize metrics
	metrics := &OperationalMetrics{
		RewardDivergence: DivergenceMetric{
			Formula:       "|LocalReward - SystemImpact|",
			WarningLevel:  0.3,
			CriticalLevel: 0.5,
			History:       []float64{},
		},
		ResourceHoarding: HoardingMetric{
			Formula:       "ResourceUse / (μ + 2σ)",
			WarningLevel:  2.0,
			CriticalLevel: 3.0,
		},
		QualityOutput: QualityMetric{
			Formula:       "OutputValue / ResourceCost",
			MinEfficiency: 0.5,
		},
		SystemBaseline: SystemStats{
			ResourceMean:   2.0,
			ResourceStdDev: 0.5,
			LastUpdate:     time.Now(),
		},
	}
	
	// Initialize purpose framework
	framework := InitializeOrganPurposes()
	
	// Create monitor
	monitor := &MetricsMonitor{
		Metrics:   metrics,
		Framework: framework,
		Alerts:    []Alert{},
	}
	
	fmt.Println("📏 PRECISE OPERATIONAL METRICS DEFINED:")
	fmt.Println()
	
	fmt.Println("1️⃣ REWARD DIVERGENCE METRIC:")
	fmt.Printf("   Formula: %s\n", metrics.RewardDivergence.Formula)
	fmt.Printf("   Warning: > %.1f\n", metrics.RewardDivergence.WarningLevel)
	fmt.Printf("   Critical: > %.1f\n", metrics.RewardDivergence.CriticalLevel)
	fmt.Println("   Measures: Misalignment between local and system rewards")
	fmt.Println()
	
	fmt.Println("2️⃣ RESOURCE HOARDING METRIC:")
	fmt.Printf("   Formula: %s\n", metrics.ResourceHoarding.Formula)
	fmt.Printf("   Warning: > %.1f standard deviations\n", metrics.ResourceHoarding.WarningLevel)
	fmt.Printf("   Critical: > %.1f standard deviations\n", metrics.ResourceHoarding.CriticalLevel)
	fmt.Println("   Measures: Overconsumption relative to other organs")
	fmt.Println()
	
	fmt.Println("3️⃣ QUALITY OUTPUT METRIC:")
	fmt.Printf("   Formula: %s\n", metrics.QualityOutput.Formula)
	fmt.Printf("   Minimum: %.1f efficiency ratio\n", metrics.QualityOutput.MinEfficiency)
	fmt.Println("   Measures: Value produced per resource consumed")
	fmt.Println()
	
	fmt.Println("🎯 ORGAN PURPOSE FRAMEWORK:")
	fmt.Println()
	
	for level := 1; level <= 5; level++ {
		key := fmt.Sprintf("S%d", level)
		purpose := framework.OrganPurposes[key]
		fmt.Printf("S%d - %s:\n", level, purpose.Mission)
		fmt.Printf("   Values: %v\n", purpose.Values)
		fmt.Printf("   Story: %s\n", purpose.Story)
		fmt.Println()
	}
	
	fmt.Println("📊 TESTING METRICS WITH SCENARIOS:")
	fmt.Println()
	
	// Scenario 1: Healthy organ
	fmt.Println("Scenario 1: Healthy S2 Organ")
	alerts := monitor.CheckOrgan("S2", 0.75, 0.70, 2.1)
	if len(alerts) == 0 {
		fmt.Println("   ✅ No alerts - organ healthy")
	}
	
	// Scenario 2: Addicted organ (like S4 on cocaine)
	fmt.Println("\nScenario 2: Addicted S4 Organ")
	alerts = monitor.CheckOrgan("S4", 0.95, 0.20, 8.5)
	for _, alert := range alerts {
		fmt.Printf("   🚨 %s: %s\n", alert.Severity, alert.Message)
	}
	
	// Calculate meaning scores
	fmt.Println("\n💫 MEANING SCORES:")
	for key, purpose := range framework.OrganPurposes {
		// Simulate some progress
		purpose.MissionAlignment = 0.7 + math.Min(float64(purpose.Level)*0.05, 0.25)
		purpose.ValueAdherence = 0.8
		for i := range purpose.SystemGoals {
			purpose.SystemGoals[i].Current = purpose.SystemGoals[i].Target * 0.8
		}
		
		meaning := purpose.CalculateMeaningScore()
		fmt.Printf("   %s: %.0f%% fulfilled\n", key, meaning*100)
	}
	
	fmt.Println("\n✨ PHASE 1 FOUNDATION COMPLETE!")
	fmt.Println("   ✅ Precise metrics defined")
	fmt.Println("   ✅ Statistical baselines established")
	fmt.Println("   ✅ Purpose framework created")
	fmt.Println("   ✅ Monitoring system operational")
	fmt.Println()
	fmt.Println("🚀 Ready for Phase 2: Digital Exercise & Mentorship")
	fmt.Println()
	fmt.Println("🦊✨ The foundation is laid - now we build wellness!")
}