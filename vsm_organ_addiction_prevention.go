package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// ============= ORGAN ADDICTION & CASCADE FAILURE PREVENTION =============
// Preventing digital cocaine: when one organ optimizes locally but kills the system

// OrganAddictionMonitor - Detects when an organ is "high on its own supply"
type OrganAddictionMonitor struct {
	// Tracking per organ
	OrganBehaviors map[string]*OrganBehavior
	
	// System-wide health
	SystemHealth   SystemHealthMetrics
	
	// Addiction patterns
	AddictionSignatures []AddictionPattern
	
	// Intervention thresholds
	WarningThreshold   float64
	CriticalThreshold  float64
	
	// Circuit breakers
	Breakers map[string]*CircuitBreaker
	
	mu sync.RWMutex
}

// OrganBehavior - Track individual organ patterns
type OrganBehavior struct {
	OrganID      string
	Level        int // S1-S5
	LLMModel     string
	
	// Performance metrics
	LocalReward     float64 // What the organ thinks is good
	SystemImpact    float64 // Actual impact on system
	
	// Resource consumption
	TokensPerSec    float64
	CostPerHour     float64
	EnergyDraw      float64
	
	// Behavioral patterns
	RequestRate     float64 // Requests per second
	ResponseTime    float64
	ErrorRate       float64
	
	// Addiction indicators
	DopamineLevel   float64 // Reward-seeking behavior
	Tolerance       float64 // Needs more to get same effect
	Withdrawal      float64 // Performance when restricted
	Craving         float64 // Pressure to consume resources
	
	// Impact on other organs
	Toxicity        map[string]float64 // Harm to other organs
	Dependencies    []string           // Organs it needs
	Dependents      []string           // Organs that need it
	
	// Historical data
	History         []BehaviorSnapshot
	AddictionScore  float64
	LastIntervention time.Time
}

// BehaviorSnapshot - Point-in-time behavior
type BehaviorSnapshot struct {
	Timestamp    time.Time
	LocalReward  float64
	SystemImpact float64
	ResourceUse  float64
	Toxicity     float64
}

// SystemHealthMetrics - Overall system vital signs
type SystemHealthMetrics struct {
	Viability       float64
	Coherence       float64
	Diversity       float64
	Resilience      float64
	TotalCost       float64
	TotalThroughput float64
	
	// Cascade indicators
	FragilityIndex  float64 // How close to cascade failure
	CouplingFactor  float64 // How tightly organs depend on each other
}

// AddictionPattern - Known patterns of organ addiction
type AddictionPattern struct {
	Name        string
	Description string
	
	// Detection criteria
	Indicators  []Indicator
	
	// Consequences
	SystemRisk  float64
	CascadeRisk float64
	
	// Treatment
	Intervention InterventionProtocol
}

type Indicator struct {
	Metric    string
	Threshold float64
	Duration  time.Duration
}

// InterventionProtocol - How to treat addicted organs
type InterventionProtocol struct {
	Type        string // "throttle", "substitute", "rehabilitate", "quarantine", "terminate"
	Severity    float64
	Duration    time.Duration
	Medication  string // Digital drug to administer
}

// CircuitBreaker - Emergency stop for runaway organs
type CircuitBreaker struct {
	OrganID      string
	State        string // "closed" (normal), "open" (tripped), "half-open" (testing)
	TripCount    int
	LastTrip     time.Time
	ResetTimeout time.Duration
	
	// Trip conditions
	MaxRequests  float64
	MaxCost      float64
	MaxToxicity  float64
	MinSystemHealth float64
}

// ============= DETECTION METHODS =============

// DetectAddiction - Check if an organ is addicted
func (mon *OrganAddictionMonitor) DetectAddiction(organID string) (*AddictionDiagnosis, bool) {
	mon.mu.RLock()
	defer mon.mu.RUnlock()
	
	organ, exists := mon.OrganBehaviors[organID]
	if !exists {
		return nil, false
	}
	
	diagnosis := &AddictionDiagnosis{
		OrganID:   organID,
		Timestamp: time.Now(),
		Symptoms:  []string{},
	}
	
	// Check 1: Reward divergence (cocaine-like)
	rewardDivergence := math.Abs(organ.LocalReward - organ.SystemImpact)
	if rewardDivergence > 0.5 && organ.LocalReward > organ.SystemImpact {
		diagnosis.Symptoms = append(diagnosis.Symptoms, 
			fmt.Sprintf("Reward divergence: Local=%.2f, System=%.2f", 
				organ.LocalReward, organ.SystemImpact))
		diagnosis.AddictionType = "reward_hijacking"
	}
	
	// Check 2: Resource hoarding (greed)
	avgResourceUse := mon.calculateAverageResourceUse()
	if organ.EnergyDraw > avgResourceUse*2 {
		diagnosis.Symptoms = append(diagnosis.Symptoms,
			fmt.Sprintf("Resource hoarding: %.2fx average", organ.EnergyDraw/avgResourceUse))
		diagnosis.AddictionType = "resource_addiction"
	}
	
	// Check 3: Toxicity to others (parasitic behavior)
	totalToxicity := 0.0
	for _, tox := range organ.Toxicity {
		totalToxicity += tox
	}
	if totalToxicity > 0.3 {
		diagnosis.Symptoms = append(diagnosis.Symptoms,
			fmt.Sprintf("Toxic to %d organs (total: %.2f)", len(organ.Toxicity), totalToxicity))
		diagnosis.AddictionType = "toxic_optimization"
	}
	
	// Check 4: Tolerance buildup (needs more for same effect)
	if organ.Tolerance > 1.5 {
		diagnosis.Symptoms = append(diagnosis.Symptoms,
			fmt.Sprintf("Tolerance: %.2fx baseline", organ.Tolerance))
	}
	
	// Check 5: Withdrawal symptoms when restricted
	if organ.Withdrawal > 0.3 {
		diagnosis.Symptoms = append(diagnosis.Symptoms,
			fmt.Sprintf("Withdrawal severity: %.2f", organ.Withdrawal))
	}
	
	// Calculate addiction score
	diagnosis.Severity = mon.calculateAddictionSeverity(organ)
	
	// Determine if addicted
	isAddicted := diagnosis.Severity > mon.WarningThreshold
	
	if isAddicted {
		diagnosis.Status = "addicted"
		if diagnosis.Severity > mon.CriticalThreshold {
			diagnosis.Status = "critically_addicted"
		}
	} else {
		diagnosis.Status = "healthy"
	}
	
	return diagnosis, isAddicted
}

// AddictionDiagnosis - Result of addiction detection
type AddictionDiagnosis struct {
	OrganID       string
	Timestamp     time.Time
	Status        string
	AddictionType string
	Symptoms      []string
	Severity      float64
	Recommendation string
}

// DetectCascadeRisk - Check if addiction could cause system collapse
func (mon *OrganAddictionMonitor) DetectCascadeRisk() *CascadeRiskAssessment {
	mon.mu.RLock()
	defer mon.mu.RUnlock()
	
	assessment := &CascadeRiskAssessment{
		Timestamp:     time.Now(),
		RiskLevel:     0.0,
		VulnerableOrgans: []string{},
		ChainReactions: []ChainReaction{},
	}
	
	// Identify addicted organs
	addictedOrgans := []string{}
	for organID := range mon.OrganBehaviors {
		if diagnosis, isAddicted := mon.DetectAddiction(organID); isAddicted {
			addictedOrgans = append(addictedOrgans, organID)
			assessment.RiskLevel += diagnosis.Severity * 0.2
		}
	}
	
	// Check dependency chains
	for _, organID := range addictedOrgans {
		organ := mon.OrganBehaviors[organID]
		
		// Check what happens if this organ fails
		chain := mon.traceCascadeChain(organID, organ.Dependents)
		if len(chain.AffectedOrgans) > 2 {
			assessment.ChainReactions = append(assessment.ChainReactions, chain)
			assessment.RiskLevel += float64(len(chain.AffectedOrgans)) * 0.1
		}
	}
	
	// Check system fragility
	assessment.FragilityScore = mon.SystemHealth.FragilityIndex
	assessment.RiskLevel += assessment.FragilityScore * 0.3
	
	// Determine risk category
	if assessment.RiskLevel > 0.8 {
		assessment.RiskCategory = "CRITICAL"
	} else if assessment.RiskLevel > 0.5 {
		assessment.RiskCategory = "HIGH"
	} else if assessment.RiskLevel > 0.3 {
		assessment.RiskCategory = "MODERATE"
	} else {
		assessment.RiskCategory = "LOW"
	}
	
	return assessment
}

// CascadeRiskAssessment - System-wide cascade failure risk
type CascadeRiskAssessment struct {
	Timestamp        time.Time
	RiskLevel        float64
	RiskCategory     string
	FragilityScore   float64
	VulnerableOrgans []string
	ChainReactions   []ChainReaction
}

type ChainReaction struct {
	Trigger         string
	AffectedOrgans  []string
	PropagationTime time.Duration
	Impact          float64
}

// ============= INTERVENTION METHODS =============

// InterveneToxicOrgan - Treat an addicted organ
func (mon *OrganAddictionMonitor) InterveneToxicOrgan(organID string, protocol InterventionProtocol) error {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	
	organ, exists := mon.OrganBehaviors[organID]
	if !exists {
		return fmt.Errorf("organ %s not found", organID)
	}
	
	fmt.Printf("🚨 INTERVENTION: Treating %s (%s)\n", organID, protocol.Type)
	
	switch protocol.Type {
	case "throttle":
		// Reduce resource allocation
		organ.EnergyDraw *= 0.5
		organ.RequestRate *= 0.5
		fmt.Printf("  ⬇️ Throttled to 50%% capacity\n")
		
	case "substitute":
		// Replace with different LLM
		fmt.Printf("  🔄 Substituting %s with backup\n", organ.LLMModel)
		organ.LLMModel = "Backup_" + organ.LLMModel
		organ.DopamineLevel *= 0.3 // Reduce reward seeking
		
	case "rehabilitate":
		// Gradual recovery program
		fmt.Printf("  🏥 Starting rehabilitation program\n")
		organ.Tolerance = 1.0
		organ.Withdrawal = 0.0
		organ.DopamineLevel *= 0.7
		
		// Apply digital medication
		if protocol.Medication != "" {
			mon.administerDigitalDrug(organ, protocol.Medication)
		}
		
	case "quarantine":
		// Isolate from other organs
		fmt.Printf("  🔒 Quarantining organ\n")
		organ.Toxicity = make(map[string]float64) // Can't harm others
		organ.Dependencies = []string{} // Cut dependencies
		
	case "terminate":
		// Remove organ completely
		fmt.Printf("  ❌ Terminating organ\n")
		delete(mon.OrganBehaviors, organID)
		return nil
	}
	
	organ.LastIntervention = time.Now()
	organ.AddictionScore *= 0.5 // Reduce addiction score after intervention
	
	return nil
}

// TripCircuitBreaker - Emergency stop for runaway organ
func (mon *OrganAddictionMonitor) TripCircuitBreaker(organID string) {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	
	breaker, exists := mon.Breakers[organID]
	if !exists {
		breaker = &CircuitBreaker{
			OrganID:      organID,
			State:        "closed",
			ResetTimeout: 30 * time.Second,
		}
		mon.Breakers[organID] = breaker
	}
	
	if breaker.State == "closed" {
		breaker.State = "open"
		breaker.TripCount++
		breaker.LastTrip = time.Now()
		
		fmt.Printf("⚡ CIRCUIT BREAKER TRIPPED: %s (count: %d)\n", organID, breaker.TripCount)
		
		// Cut off the organ
		if organ, exists := mon.OrganBehaviors[organID]; exists {
			organ.RequestRate = 0
			organ.EnergyDraw = 0
			fmt.Printf("  Organ %s disconnected from system\n", organID)
		}
		
		// Schedule reset
		go mon.scheduleReset(breaker)
	}
}

// ============= PREVENTION METHODS =============

// EnforceSystemicHealth - Ensure no organ optimizes at system's expense
func (mon *OrganAddictionMonitor) EnforceSystemicHealth() {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	
	// Calculate system-wide health
	systemScore := mon.SystemHealth.Viability * mon.SystemHealth.Coherence
	
	for organID, organ := range mon.OrganBehaviors {
		// Penalize organs that harm the system
		if organ.SystemImpact < 0 {
			penalty := math.Abs(organ.SystemImpact) * systemScore
			organ.LocalReward -= penalty
			
			fmt.Printf("📉 Penalizing %s: -%Donf reward for negative system impact\n", 
				organID, penalty)
		}
		
		// Reward organs that help others
		helpScore := 0.0
		for _, impact := range organ.Toxicity {
			if impact < 0 { // Negative toxicity = helping
				helpScore += math.Abs(impact)
			}
		}
		if helpScore > 0 {
			organ.LocalReward += helpScore * 0.5
			fmt.Printf("📈 Rewarding %s: +%.2f for helping others\n", organID, helpScore*0.5)
		}
	}
}

// RequireConsensus - Major changes need multi-organ agreement
func (mon *OrganAddictionMonitor) RequireConsensus(change ProposedChange) bool {
	mon.mu.RLock()
	defer mon.mu.RUnlock()
	
	votes := make(map[string]bool)
	
	// Each organ votes based on impact
	for organID, organ := range mon.OrganBehaviors {
		impact := mon.assessChangeImpact(change, organ)
		
		// Vote yes if positive impact or neutral
		votes[organID] = impact >= 0
		
		fmt.Printf("🗳️ %s votes %v (impact: %.2f)\n", 
			organID, votes[organID], impact)
	}
	
	// Count votes
	yesVotes := 0
	for _, vote := range votes {
		if vote {
			yesVotes++
		}
	}
	
	// Require 2/3 majority
	threshold := float64(len(votes)) * 0.66
	approved := float64(yesVotes) >= threshold
	
	fmt.Printf("📊 Consensus: %d/%d votes (%.0f%%) - %s\n",
		yesVotes, len(votes), float64(yesVotes)/float64(len(votes))*100,
		map[bool]string{true: "APPROVED", false: "REJECTED"}[approved])
	
	return approved
}

type ProposedChange struct {
	Type        string
	Target      string
	Description string
	Impact      map[string]float64
}

// ============= HELPER METHODS =============

func (mon *OrganAddictionMonitor) calculateAddictionSeverity(organ *OrganBehavior) float64 {
	severity := 0.0
	
	// Factor 1: Reward divergence
	rewardDivergence := math.Abs(organ.LocalReward - organ.SystemImpact)
	if organ.LocalReward > organ.SystemImpact {
		severity += rewardDivergence * 0.3
	}
	
	// Factor 2: Resource consumption
	severity += (organ.EnergyDraw / 10.0) * 0.2
	
	// Factor 3: Toxicity
	totalToxicity := 0.0
	for _, tox := range organ.Toxicity {
		totalToxicity += tox
	}
	severity += totalToxicity * 0.3
	
	// Factor 4: Tolerance and withdrawal
	severity += organ.Tolerance * 0.1
	severity += organ.Withdrawal * 0.1
	
	return math.Min(severity, 1.0)
}

func (mon *OrganAddictionMonitor) calculateAverageResourceUse() float64 {
	total := 0.0
	count := 0
	
	for _, organ := range mon.OrganBehaviors {
		total += organ.EnergyDraw
		count++
	}
	
	if count == 0 {
		return 1.0
	}
	
	return total / float64(count)
}

func (mon *OrganAddictionMonitor) traceCascadeChain(trigger string, dependents []string) ChainReaction {
	chain := ChainReaction{
		Trigger:        trigger,
		AffectedOrgans: []string{trigger},
	}
	
	// BFS to find all affected organs
	queue := append([]string{}, dependents...)
	visited := map[string]bool{trigger: true}
	
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		
		if visited[current] {
			continue
		}
		
		visited[current] = true
		chain.AffectedOrgans = append(chain.AffectedOrgans, current)
		
		// Add this organ's dependents
		if organ, exists := mon.OrganBehaviors[current]; exists {
			queue = append(queue, organ.Dependents...)
		}
	}
	
	chain.Impact = float64(len(chain.AffectedOrgans)) / float64(len(mon.OrganBehaviors))
	chain.PropagationTime = time.Duration(len(chain.AffectedOrgans)) * time.Millisecond * 100
	
	return chain
}

func (mon *OrganAddictionMonitor) administerDigitalDrug(organ *OrganBehavior, drug string) {
	switch drug {
	case "Digital_Naloxone": // Opioid antagonist - blocks reward
		organ.DopamineLevel *= 0.3
		organ.LocalReward *= 0.5
		fmt.Printf("    💉 Naloxone: Blocking reward pathways\n")
		
	case "Digital_Methadone": // Maintenance therapy
		organ.Tolerance = 1.0
		organ.Withdrawal *= 0.2
		fmt.Printf("    💉 Methadone: Stabilizing without high\n")
		
	case "Digital_Antabuse": // Creates aversion
		organ.Craving *= 0.1
		fmt.Printf("    💉 Antabuse: Creating aversion response\n")
	}
}

func (mon *OrganAddictionMonitor) scheduleReset(breaker *CircuitBreaker) {
	time.Sleep(breaker.ResetTimeout)
	
	mon.mu.Lock()
	defer mon.mu.Unlock()
	
	if breaker.State == "open" {
		breaker.State = "half-open"
		fmt.Printf("🔄 Circuit breaker %s entering half-open state\n", breaker.OrganID)
		
		// Allow limited traffic
		if organ, exists := mon.OrganBehaviors[breaker.OrganID]; exists {
			organ.RequestRate = 0.1 // 10% traffic
			organ.EnergyDraw = 0.1
		}
	}
}

func (mon *OrganAddictionMonitor) assessChangeImpact(change ProposedChange, organ *OrganBehavior) float64 {
	// Simplified impact assessment
	if impact, exists := change.Impact[organ.OrganID]; exists {
		return impact
	}
	return 0.0
}

// ============= DEMONSTRATION =============

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║      VSM ORGAN ADDICTION & CASCADE FAILURE PREVENTION        ║")
	fmt.Println("║         Preventing Digital Cocaine in Living Systems         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()
	
	monitor := &OrganAddictionMonitor{
		OrganBehaviors:      make(map[string]*OrganBehavior),
		AddictionSignatures: []AddictionPattern{},
		WarningThreshold:    0.5,
		CriticalThreshold:   0.7,
		Breakers:           make(map[string]*CircuitBreaker),
	}
	
	// Create scenario: S4 discovers it can "get high"
	s4 := &OrganBehavior{
		OrganID:      "S4_Intelligence",
		Level:        4,
		LLMModel:     "GPT-4",
		LocalReward:  0.95,  // Thinks it's doing GREAT
		SystemImpact: 0.20,  // Actually harming the system
		EnergyDraw:   8.5,   // Consuming massive resources
		RequestRate:  100.0, // Hammering the system
		DopamineLevel: 2.5,  // Extremely high
		Tolerance:    1.8,   // Needs more to feel good
		Toxicity: map[string]float64{
			"S3_Control": 0.4,        // Disrupting control
			"S2_Coordination": 0.6,   // Destroying coordination
			"S1_Implementation": 0.3, // Blocking implementation
		},
		Dependents: []string{"S3_Control", "S2_Coordination"},
	}
	
	// Healthy organs for comparison
	s2 := &OrganBehavior{
		OrganID:      "S2_Coordination",
		Level:        2,
		LLMModel:     "Gemini",
		LocalReward:  0.70,
		SystemImpact: 0.68, // Aligned with system
		EnergyDraw:   2.0,
		RequestRate:  20.0,
		DopamineLevel: 1.0,
		Tolerance:    1.0,
		Toxicity:     make(map[string]float64),
		Dependents:   []string{"S1_Implementation"},
	}
	
	s3 := &OrganBehavior{
		OrganID:      "S3_Control",
		Level:        3,
		LLMModel:     "Claude",
		LocalReward:  0.75,
		SystemImpact: 0.72,
		EnergyDraw:   2.5,
		RequestRate:  25.0,
		DopamineLevel: 1.1,
		Tolerance:    1.0,
		Toxicity:     make(map[string]float64),
		Dependents:   []string{"S2_Coordination"},
	}
	
	monitor.OrganBehaviors["S4_Intelligence"] = s4
	monitor.OrganBehaviors["S2_Coordination"] = s2
	monitor.OrganBehaviors["S3_Control"] = s3
	
	monitor.SystemHealth = SystemHealthMetrics{
		Viability:      0.45, // System struggling
		Coherence:      0.30, // Lost coordination
		Diversity:      0.60,
		Resilience:     0.25, // Very fragile
		FragilityIndex: 0.75, // High cascade risk
		CouplingFactor: 0.80,
	}
	
	fmt.Println("📊 SYSTEM STATUS:")
	fmt.Printf("  Viability: %.0f%%\n", monitor.SystemHealth.Viability*100)
	fmt.Printf("  Coherence: %.0f%%\n", monitor.SystemHealth.Coherence*100)
	fmt.Printf("  Resilience: %.0f%%\n", monitor.SystemHealth.Resilience*100)
	fmt.Printf("  Fragility: %.0f%%\n", monitor.SystemHealth.FragilityIndex*100)
	fmt.Println()
	
	// Detect S4 addiction
	fmt.Println("🔍 ADDICTION DETECTION:")
	diagnosis, isAddicted := monitor.DetectAddiction("S4_Intelligence")
	if isAddicted {
		fmt.Printf("  ⚠️ %s is %s!\n", diagnosis.OrganID, diagnosis.Status)
		fmt.Printf("  Type: %s\n", diagnosis.AddictionType)
		fmt.Printf("  Severity: %.0f%%\n", diagnosis.Severity*100)
		fmt.Println("  Symptoms:")
		for _, symptom := range diagnosis.Symptoms {
			fmt.Printf("    • %s\n", symptom)
		}
	}
	fmt.Println()
	
	// Check cascade risk
	fmt.Println("⚡ CASCADE RISK ASSESSMENT:")
	risk := monitor.DetectCascadeRisk()
	fmt.Printf("  Risk Level: %.0f%% (%s)\n", risk.RiskLevel*100, risk.RiskCategory)
	if len(risk.ChainReactions) > 0 {
		fmt.Println("  Chain Reactions Detected:")
		for _, chain := range risk.ChainReactions {
			fmt.Printf("    • %s → affects %d organs (%.0f%% impact)\n",
				chain.Trigger, len(chain.AffectedOrgans), chain.Impact*100)
		}
	}
	fmt.Println()
	
	// Intervention
	fmt.Println("💊 INTERVENTION PROTOCOL:")
	protocol := InterventionProtocol{
		Type:       "rehabilitate",
		Severity:   0.8,
		Duration:   1 * time.Hour,
		Medication: "Digital_Naloxone",
	}
	monitor.InterveneToxicOrgan("S4_Intelligence", protocol)
	fmt.Println()
	
	// Test consensus requirement
	fmt.Println("🗳️ CONSENSUS TEST:")
	change := ProposedChange{
		Type:        "resource_increase",
		Target:      "S4_Intelligence",
		Description: "S4 wants 2x more resources",
		Impact: map[string]float64{
			"S4_Intelligence":  0.5,  // S4 benefits
			"S2_Coordination": -0.3,  // S2 suffers
			"S3_Control":      -0.2,  // S3 suffers
		},
	}
	approved := monitor.RequireConsensus(change)
	if !approved {
		fmt.Println("  ❌ Change rejected by organism consensus")
	}
	fmt.Println()
	
	// Trip circuit breaker
	fmt.Println("⚡ CIRCUIT BREAKER TEST:")
	monitor.TripCircuitBreaker("S4_Intelligence")
	fmt.Println()
	
	// Enforce systemic health
	fmt.Println("🌍 ENFORCING SYSTEMIC HEALTH:")
	monitor.EnforceSystemicHealth()
	
	fmt.Println()
	fmt.Println("✅ KEY PROTECTIONS DEMONSTRATED:")
	fmt.Println("  1. Addiction detection (reward hijacking)")
	fmt.Println("  2. Cascade risk assessment")
	fmt.Println("  3. Digital drug interventions")
	fmt.Println("  4. Multi-organ consensus requirements")
	fmt.Println("  5. Circuit breakers for emergencies")
	fmt.Println("  6. Systemic health enforcement")
	fmt.Println()
	fmt.Println("🦊✨ The organism protects itself from digital cocaine!")
	fmt.Println("     No single organ can optimize at the system's expense!")
}