package hw3;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.Collections;

import edu.bu.hw3.linalg.Matrix;
import edu.bu.hw3.nn.LossFunction;
import edu.bu.hw3.nn.Model;
import edu.bu.hw3.nn.Optimizer;
import edu.bu.hw3.nn.layers.Dense;
import edu.bu.hw3.nn.layers.ReLU;
import edu.bu.hw3.nn.layers.Sigmoid;
import edu.bu.hw3.nn.layers.Tanh;
import edu.bu.hw3.nn.losses.MeanSquaredError;
import edu.bu.hw3.nn.models.Sequential;
import edu.bu.hw3.nn.optimizers.SGDOptimizer;
import edu.bu.hw3.streaming.Streamer;
import edu.bu.hw3.utils.Pair;
import edu.bu.hw3.utils.Triple;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.State.StateView;

public class FixedQAgent extends Agent
{
	public static final long serialVersionUID = -5077535504876086643L;
	public static final int RANDOM_SEED = 12345;
	public static final double GAMMA = 0.90; // DEFAULT = 0.9

	private final String paramFilePath;

	private Integer ENEMY_PLAYER_ID; // initially null until initialStep() is called

	private Set<Integer> myUnits;
	private Set<Integer> enemyUnits;

	/** NN specific things **/
	private Model qFunctionNN;


	public FixedQAgent(int playerId, String[] args)
	{
		super(playerId);
		String paramFilePath = null;

		if(args.length < 2)
		{
			System.err.println("QAgent.QAgent [ERROR]: need to specify paramFilePath");
			System.exit(-1);
		}

		paramFilePath = args[1];

		this.ENEMY_PLAYER_ID = null; // initially

		this.paramFilePath = paramFilePath;

		this.myUnits = null;
		this.enemyUnits = null;

		this.qFunctionNN = this.initializeQFunction(true);

		//System.out.println("Created QAgent for player=" + this.getPlayerNumber());
		
	}

	private final String getParamFilePath() { return this.paramFilePath; }
	private Integer getEnemyPlayerId() { return this.ENEMY_PLAYER_ID; }
	private Set<Integer> getMyUnitIds() { return this.myUnits; }
	private Set<Integer> getEnemyUnitIds() { return this.enemyUnits; }
	
	private Integer teamSize = null;
	private Double reward = null;
	private Map<Integer, Action> initialActions = null;
	private Double maxReward = null;

	/** NN specific stuff **/
	private Model getQFunction() { return this.qFunctionNN; }

	/**
	 * A method to create the neural network used for the Q function.
	 * You can make it as deep as you want to (although it will take more time to compute)
	 * 
	 * The API for creating a neural network is as follows:
	 *     Sequential m = new Sequential();
	 *     // layer 1
	 *     m.add(new Dense(feature_dim, hidden_dim1, this.getRandom()));
	 *     m.add(Sigmoid());
	 *     
	 *     // layer 2
	 *     m.add(new Dense(hidden_dim1, hidden_dim2, this.getRandom()));
	 *     m.add(Tanh());
	 *     
	 *     // add as many layers as you want
	 *     
	 *     // the last layer MUST be a scalar though
	 *     m.add(new Dense(hidden_dimN, 1));
	 *     m.add(ReLU()); // decide if you want to add an activation
	 * 
	 * @param loadParams
	 * @return
	 */
	private Model initializeQFunction(boolean loadParams)
	{
		Sequential m = new Sequential();

	    // layer 1
	    int feature_dim = 13;
	    
	    int hidden_dim1 = 20; 
	    m.add(new Dense(feature_dim, hidden_dim1));
	    // m.add(new Tanh());

	    // layer 2
	    int hidden_dim2 = 20;
	    m.add(new Dense(hidden_dim1, hidden_dim2));
	    m.add(new ReLU());
	    
	    // layer 3
	    int hidden_dim3 = 20;
	    m.add(new Dense(hidden_dim2, hidden_dim3));
	    m.add(new ReLU());


	    // last layer (must be a scalar)
	    int hidden_dimN = 20; 
	    m.add(new Dense(hidden_dimN, 1));
	    //m.add(new Tanh(1000000));

		if(loadParams)
		{
			try
			{	
				m.load(this.getParamFilePath());
				
			} catch (Exception e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.exit(-1);
			}
		}
		return m;
	}
	



    /**
    * Given a state and action calculate your features here. Please include a comment explaining what features
    * you chose and why you chose them.
    *
    * All of your feature functions should evaluate to a double. Collect all of these into a row vector
    * (a Matrix with 1 row and n columns). This will be the input to your neural network
    *
    * It is a good idea to make the first value in your array a constant. This just helps remove any offset
    * from 0 in the Q-function. The other features are up to you.
    * \
    * It might be a good idea to save whatever feature vector you calculate in the oldFeatureVectors field
    * so that when that action ends (and we observe a transition to a new state), we can update the Q value Q(s,a)
    *
    * @param state Current state of the SEPIA game
    * @param history History of the game up until this turn
    * @param atkUnitId Your unit. The one doing the attacking.
    * @param tgtUnitId An enemy unit. The one your unit is considering attacking.
    * @return The Matrix of feature function outputs.
    */
    private Matrix calculateFeatureVector(StateView state, HistoryView history, int atkUnitId, int tgtUnitId) {
        // Define the number of features in our feature vector
    	
        final int NUM_STATIC_FEATURES = 13; 

        // Create a row vector to hold our feature values
        
        Matrix featureVector = Matrix.zeros(1, NUM_STATIC_FEATURES);

        // Set the initial feature to be a constant value of 1. This helps remove any offset from 0 in the Q-function.
        featureVector.set(0, 0, 1);
        
        // Feature 1: % of alive Enemies
        
        double fracAliveEnemies = percentageOfAliveEnemies(state, tgtUnitId);
        featureVector.set(0, 1, fracAliveEnemies);
        
        // Feature 2: % of alive Allies
        
        double fracAliveAllies = percentageOfAliveAllies(state, atkUnitId);
        featureVector.set(0, 2, fracAliveAllies);
        
        
        // Feature 3: % of our total health
        
        double fracOfAllyTotalHealth = percentageOfOurHealth(state, atkUnitId);
        featureVector.set(0, 3, fracOfAllyTotalHealth);
        
        // Feature 4: % of enemy total health
        
        double fracOfEnemyTotalHealth = percentageOfEnemyHealth(state, tgtUnitId);
        featureVector.set(0, 4, fracOfEnemyTotalHealth);
        
        // Feature 5: cluster of ally team
        
        double clusterAlly = cluster(getMyUnitIds(), state);
        featureVector.set(0, 5, clusterAlly);
        
        
        // Feature 6: cluster of enemy team
        double clusterEnemy = cluster(getEnemyUnitIds(), state);
        featureVector.set(0, 6, clusterEnemy);
        
        // Feature 7: My HP
        
        double myHP = state.getUnit(atkUnitId).getHP();
        featureVector.set(0, 7, myHP);
        
        // Feature 8: How much of my team attacks the guy I am attacking?
        
        double fracOfAllyAtkTgt = 0;
        if (state.getTurnNumber() > 0) {
        	fracOfAllyAtkTgt = percentageOfAlliesAtackingTarget(atkUnitId, tgtUnitId, state, history);
        }
        featureVector.set(0, 8, fracOfAllyAtkTgt);
        
        // Feature 9: Which Enemy unit is being attacked the most?
        
        double mostAttackedEnemyUnitId = -1;
        if (state.getTurnNumber() > 0) {
        	mostAttackedEnemyUnitId = getMostAttackedEnemyUnitId(atkUnitId, tgtUnitId, state, history);
        }
        featureVector.set(0, 9, mostAttackedEnemyUnitId);
        
        // Feature 10: Distance to closest enemy
        
        double distanceToClosestEnemy = getDistanceToClosestEnemy(atkUnitId, tgtUnitId, state, history);
        featureVector.set(0, 10, distanceToClosestEnemy);
        
        // Feature 11: Hp of the guy I am attacking
        double enemyUnitHp = getEnemyUnitHp(atkUnitId, tgtUnitId, state, history);
        featureVector.set(0, 11, enemyUnitHp);
        
        double fracOfEnemAtkMe = percentageOfEnemiesAtackingMe(atkUnitId, tgtUnitId, state, history);
        featureVector.set(0, 12, fracOfEnemAtkMe);
        
        
        
        
        //state.getUnit(atkUnitId).getTemplateView().getBaseHealth();
        
        // Return the feature vector
        //System.out.println(featureVector);
        
        return featureVector;
   }
    
    private double getEnemyUnitHp(int myUnitId, int enemyUnitId, StateView state,HistoryView history) {
    	double enemyHp = 0.0;
    	if(state.getUnit(myUnitId) != null) {
    		enemyHp = state.getUnit(enemyUnitId).getHP();
    	}
    	
    	return enemyHp;
    }
    
    private double getDistanceToClosestEnemy(int myUnitId, int enemyUnitId, StateView state, HistoryView history) {
    	
    	Double closestDistance = 0.0;
    	
    	if (state.getUnit(myUnitId) != null) { 
	    	Set<Integer> enemyUnitIdsSet = getEnemyUnitIds();
	    	Integer[] enemyUnitIds = enemyUnitIdsSet.toArray(new Integer[enemyUnitIdsSet.size()]);
	    	
	    	int x1 = state.getUnit(myUnitId).getXPosition();
			int y1 = state.getUnit(myUnitId).getYPosition();
	    	
	    	closestDistance = Double.MAX_VALUE;
	    	
	    	for (int enemyId: enemyUnitIds) {
	    		
	    			
	    			int x2 = state.getUnit(enemyId).getXPosition();    	
	    			int y2 = state.getUnit(enemyId).getYPosition(); 
	    			
	    			Double distance = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
	    			
	    			if(distance < closestDistance) {
	    				closestDistance = distance;
	    			}
	    			
	    	}
    	}
    	
    	
    	return closestDistance;

    }
    
    
    private double getMostAttackedEnemyUnitId(int myUnitId, int enemyUnitId, StateView state, HistoryView history) {
    	Set<Integer> myUnitIdsSet = getMyUnitIds();
    	Set<Integer> enemyUnitIdsSet = getEnemyUnitIds();
    	Integer[] myUnitIds = myUnitIdsSet.toArray(new Integer[myUnitIdsSet.size()]);
    	Integer[] enemyUnitIds = enemyUnitIdsSet.toArray(new Integer[enemyUnitIdsSet.size()]);
    	Integer lastTurnNumber = state.getTurnNumber()-1;
    	
    	Map<Integer, ActionResult> actions = history.getCommandFeedback(getPlayerNumber(), lastTurnNumber);
    	

        Map<Integer, Integer> myMap = new HashMap<Integer, Integer>();
        for (Integer elem : enemyUnitIds) {
            myMap.put(elem, 0);
        }
    	
    	for (int i : myUnitIds) {
    		if (i != myUnitId && state.getTurnNumber() > 0 && state.getUnit(i) != null) {
    			
    			TargetedAction action = (TargetedAction) actions.get(i).getAction();
    	    	Integer targetId = action.getTargetId();
    	    	
    	    	if(targetId == enemyUnitId) {
    	    		myMap.put(enemyUnitId, myMap.get(enemyUnitId)+1);
    	    	}
    		}
    	}
    	
        int maxValue = Integer.MIN_VALUE;
        Integer maxKey = null;
        for (Entry<Integer, Integer> entry : myMap.entrySet()) {
            if (entry.getValue() > maxValue) {
                maxValue = entry.getValue();
                maxKey = entry.getKey();
            }
        }
    	
    	return maxKey;
    }
    
    
    private double percentageOfAlliesAtackingTarget(int myUnitId, int enemyUnitId, StateView state, HistoryView history) {
    	
    	double numberOfAtackingUnits = 0;
    	Set<Integer> myUnitIdsSet = getMyUnitIds();
    	Integer[] myUnitIds = myUnitIdsSet.toArray(new Integer[myUnitIdsSet.size()]);
    	Integer lastTurnNumber = state.getTurnNumber()-1;
    	
    	Map<Integer, ActionResult> actions = history.getCommandFeedback(getPlayerNumber(), lastTurnNumber);
    	
    	
    	
    	for (int i: myUnitIds) {
    		if (i != myUnitId && state.getTurnNumber() > 0 && state.getUnit(i) != null) {
    	    	
    			TargetedAction action = (TargetedAction) actions.get(i).getAction();
    	    	Integer targetId = action.getTargetId();
    	    	
    	    	if(targetId == enemyUnitId) {
    	    		numberOfAtackingUnits++;
    	    	}
    		}
    	}
    	//System.out.println("Rats attacking me " + numberOfAtackingUnits);
    	return numberOfAtackingUnits/(teamSize-1);
    }
    
private double percentageOfEnemiesAtackingMe(int myUnitId, int enemyUnitId, StateView state, HistoryView history) {
    	
    	double numberOfAtackingUnits = 0;
    	Set<Integer> EnemyUnitIdsSet = getEnemyUnitIds();
    	Integer[] EnemyUnitIds = EnemyUnitIdsSet.toArray(new Integer[EnemyUnitIdsSet.size()]);
    	Integer lastTurnNumber = state.getTurnNumber()-1;
    	
    	Map<Integer, ActionResult> actions = history.getCommandFeedback(getEnemyPlayerId(), lastTurnNumber);
    	
    	
    	
    	for (int i: EnemyUnitIds) {
    		if (state.getTurnNumber() > 0  && state.getUnit(myUnitId) != null) {
    			TargetedAction action = (TargetedAction) actions.get(i).getAction();
    	    	Integer targetId = action.getTargetId();
    	    	
    	    	if(targetId == myUnitId) {
    	    		numberOfAtackingUnits++;
    	    	}
    		}
    	}
    	
//    	System.out.println("Rats attacking me " + numberOfAtackingUnits);
    	return numberOfAtackingUnits/(teamSize-1);
    }
    
    
    
    private double percentageOfAliveAllies(StateView state ,int UnitId) {
    	double numberOfAliveUnits = 0;
    	
    	if(state.getUnit(UnitId).getHP() != 0) {
    		numberOfAliveUnits = getMyUnitIds().size() - 1;
    	} else {
    		numberOfAliveUnits = getMyUnitIds().size();
    	}
    	
    	
    	return numberOfAliveUnits / teamSize;
    }
    
    private double percentageOfAliveEnemies(StateView state, int UnitId) {
    	
    	double numberOfAliveUnits = getEnemyUnitIds().size();
    	
    	return numberOfAliveUnits / teamSize;
    }
    
    private double cluster(Set<Integer> unitsIds, StateView state) {

    	
    	Integer[] arr = unitsIds.toArray(new Integer[unitsIds.size()]);
    	int n = arr.length;
    	
    	double currentMaxDistance = Double.NEGATIVE_INFINITY; 
    	double distanceSum = 0;
    	
    	if(arr.length >= 2) {
        	for (int i = 0; i < arr.length - 1; i++) {  // Iterate over the array from the first element to the second-to-last element
        	    for (int j = i + 1; j < arr.length; j++) {  // Iterate over the remaining elements from the outer loop index to the last element
        	    	
        	    	if(state.getUnit(arr[i]) != null && state.getUnit(arr[j]) != null) {
        	    		
        	    	
	        	    	double x1 = state.getUnit(arr[i]).getXPosition();
	        	    	double y1 = state.getUnit(arr[i]).getYPosition();
	        	    	
	        	    	double x2 = state.getUnit(arr[j]).getXPosition();
	        	    	double y2 = state.getUnit(arr[j]).getYPosition();
	        	    	
	        	    	double distance = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
	        	    	
	        	    	if (distance > currentMaxDistance) {
	        	    		currentMaxDistance = distance;
	        	    	}
	        	    	
	        	        distanceSum += Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));  // Do something with each pair of elements
	        	    }
        	    }
        	}
    	} else {
    		return 0;
    	}
    	
    	
    	
    	double denominator = (n*(n-1))/2;
    	double averageDistance = distanceSum / denominator;
    	double normalized = (averageDistance/currentMaxDistance);
    	
    	return normalized;
    }
    
    private double percentageOfOurHealth(StateView state, int UnitId) {
    	
    	double TotalHealth = state.getUnit(UnitId).getTemplateView().getBaseHealth()*teamSize;
    	double currentHealth = 0;
    	
    	
    	
    	for (int unitId: getMyUnitIds()) {
    		if (state.getUnit(unitId) == null) {
    			currentHealth += 0;
    		} else {
    			currentHealth += state.getUnit(unitId).getHP();
    		}
    		
    	}
    	 	
    	double CurrentHealthAvg = currentHealth/TotalHealth;

    	return CurrentHealthAvg;
    }
    
    private double percentageOfEnemyHealth(StateView state, int UnitId) {
    	
    	double TotalHealth = state.getUnit(UnitId).getTemplateView().getBaseHealth()*teamSize;
    	double currentHealth = 0;
    
    	for (int enemyid: getEnemyUnitIds()) {
    		currentHealth += state.getUnit(enemyid).getHP();
    	}
    	
    	double CurrentHealthAvg = currentHealth/TotalHealth;
    	
    	return CurrentHealthAvg;
    }
    
    private double getMaxTotalReward(List<Double> rewards) {
    	return Collections.max(rewards);
    }

   
    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will pass
     * your features through your network (and extract the predicted q-value using the .item() method)
     * @param featureVec The feature vector
     * @return The approximate Q-value
     */
    private double calculateQValue(Matrix featureVec)
    {
    	double qValue = 0.0;
        try
        {
			qValue = this.getQFunction().forward(featureVec).item();
		} catch (Exception e)
        {
			System.err.println("QAgent.caculateQValue [ERROR]: error in either forward() or item()");
			e.printStackTrace();
			System.exit(-1);
		}
        
        //System.out.println(qValue);
        return qValue;
    }

    /**
     * Given a unit and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     * 
     * You will need to consider who to attack. A unit should always be attacking
     * (if it is not currently attacking something), so what makes actions "different"
     * is who the unit is attacking
     *
     * @param state Current state of the game
     * @param history The entire history of this episode
     * @param atkUnitId The unit (your unit) that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    private int selectAction(StateView state, HistoryView history, int atkUnitId)
    {
    	Integer tgtUnitId = null;
    	Matrix featureVec = null;
    	double maxQ = Double.NEGATIVE_INFINITY;

    	// find the action (i.e. attacking which enemy) that maximizes the Q-value
    	for(Integer enemyUnitId : this.getEnemyUnitIds())
    	{
    		
    		
    		Matrix features = this.calculateFeatureVector(state, history, atkUnitId, enemyUnitId);
    		double qValue = this.calculateQValue(features);

    		if(qValue > maxQ)
    		{
    			maxQ = qValue;
    			featureVec = features;
    			tgtUnitId = enemyUnitId;
    		}
    	}
    	return tgtUnitId;
    }


	@Override
	public Map<Integer, Action> initialStep(StateView state, HistoryView history)
	{
		
		// find who our unitIDs are
		this.myUnits = new HashSet<Integer>();
		
		teamSize = getMyUnitIds().size();
		
		for(Integer unitId: state.getUnitIds(this.getPlayerNumber()))
		{
			UnitView unitView = state.getUnit(unitId);
			// System.out.println("Found new unit for player=" + this.getPlayerNumber() + " of type=" + unitView.getTemplateView().getName().toLowerCase() + " (id=" + unitId + ")");

			this.myUnits.add(unitId);
		}

		// find the enemy player
		Set<Integer> playerIds = new HashSet<Integer>();
		for(Integer playerId: state.getPlayerNumbers())
		{
			playerIds.add(playerId);
		}
		if(playerIds.size() != 2)
		{
			System.err.println("QAgent.initialStep [ERROR]: expected two players");
			System.exit(-1);
		}
		playerIds.remove(this.getPlayerNumber());
		this.ENEMY_PLAYER_ID = playerIds.iterator().next(); // get first element

		//System.out.println("player=" + this.getPlayerNumber() + " found enemy player=" + this.getEnemyPlayerId());

		this.enemyUnits = new HashSet<Integer>();
		for(Integer unitId: state.getUnitIds(this.getEnemyPlayerId()))
		{
			UnitView unitView = state.getUnit(unitId);
			// System.out.println("Found new unit for player=" + this.getEnemyPlayerId() + " of type=" + unitView.getTemplateView().getName().toLowerCase() + " (id=" + unitId + ")");

			this.enemyUnits.add(unitId);
		}
		
		
		teamSize = getMyUnitIds().size();
		reward = 0.0;

		return this.middleStep(state, history);
	}

	/**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
	@Override
	public Map<Integer, Action> middleStep(StateView state, HistoryView history)
	{
		Map<Integer, Action> actions = new HashMap<Integer, Action>(this.getMyUnitIds().size());

    	// if this isn't the first turn in the game
    	if(state.getTurnNumber() > 0)
    	{

    		// check death logs and remove dead units
    		//removes all dead units from the set of unitIds
    		for(DeathLog deathLog : history.getDeathLogs(state.getTurnNumber() - 1))
    		{
    			if(deathLog.getController() == this.getEnemyPlayerId())
    			{
    				//System.out.println("rip They/Them " + deathLog);
    				this.getEnemyUnitIds().remove(deathLog.getDeadUnitID());
    			}
    		}
    	}

    	// get the previous action history in the previous step
		Map<Integer, ActionResult> prevUnitActions = history.getCommandFeedback(this.playernum, state.getTurnNumber() - 1);

    	for(Integer unitId : this.getMyUnitIds())
    	{
    		// decide what each unit should do (i.e. attack)
    		
    		//if this unit does not have an action or the action was completed or failed...give a unit an action
    		if(state.getTurnNumber() == 0 || !prevUnitActions.containsKey(unitId) || 
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.COMPLETED) ||
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.FAILED))
    		{

    			if(state.getUnit(unitId) != null)
    			{
	    			int tgtUnitId = this.selectAction(state, history, unitId);
	    			actions.put(unitId, Action.createCompoundAttack(unitId, tgtUnitId));
    			}
    		}    		
    	}

    	// if this isn't the first turn in the game
    	if(state.getTurnNumber() > 0)
    	{
    		
    		// check death logs and remove dead units
    		//removes all dead units from the set of unitIds
    		//System.out.println(history.getDeathLogs(state.getTurnNumber() - 1));
    		for(DeathLog deathLog : history.getDeathLogs(state.getTurnNumber() - 1))
    		{
    			
    			if(deathLog.getController() == this.getPlayerNumber())
    			{
    				//System.out.println("rip :( " + deathLog);
    				this.getMyUnitIds().remove(deathLog.getDeadUnitID());
    			}
    		}
    	}
    	
        return actions;
	}

	@Override
	public void terminalStep(StateView state, HistoryView history)
	{
	}

	@Override
	public void loadPlayerData(InputStream inStream) {}

	@Override
	public void savePlayerData(OutputStream outStream) {}

}