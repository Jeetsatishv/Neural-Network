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

public class QAgent extends Agent
{
	public static final long serialVersionUID = -5077535504876086643L;
	public static final int RANDOM_SEED = 12345;
	public static final double GAMMA = 0.90; // DEFAULT = 0.9
	public static final double LEARNING_RATE = 0.0001; // DEFAULT = 0.0001
	public static final double EPSILON = 0.02; // prob of ignoring the policy and choosing a random action

	// our agent will play this many training episodes in a row before testing
	public static final int NUM_TRAINING_EPISODES_IN_BATCH = 10;

	// our agent will play this many testing episodes in a row before training again
	public static final int NUM_TESTING_EPISODES_IN_BATCH = 5;

	private final String paramFilePath;

	private Streamer streamer; 

	private final int NUM_EPISODES_TO_PLAY;

	private int numTestEpisodesPlayedInBatch = -1;
	private int numTrainingEpisodesPlayed = 0;

	// rng to keep things repeatable...will combine with the RANDOM_SEED
	public final Random random;

	private Integer ENEMY_PLAYER_ID; // initially null until initialStep() is called

	private Set<Integer> myUnits;
	private Set<Integer> enemyUnits;
	private List<Double> totalRewards;

	/** NN specific things **/
	private Model qFunctionNN;
	private LossFunction lossFunction;
	private Optimizer optimizer;

	// how we remember what was the state, Q-value, and reward from the past
	private Map<Integer, Triple<Matrix, Matrix, Double> > oldInfoPerUnit;

	public QAgent(int playerId, String[] args)
	{
		super(playerId);
		String streamerArgString = null;
		String paramFilePath = null;

		if(args.length < 3)
		{
			System.err.println("QAgent.QAgent [ERROR]: need to specify playerId, streamerArgString, paramFilePath");
			System.exit(-1);
		}

		streamerArgString = args[1];
		paramFilePath = args[2];

		int numEpisodesToPlay = QAgent.NUM_TRAINING_EPISODES_IN_BATCH;
		boolean loadParams = false;
		if(args.length >= 4)
		{
			numEpisodesToPlay = Integer.parseInt(args[3]);
			if(args.length >= 5)
			{
				loadParams = Boolean.parseBoolean(args[4]);
			}
		}

		this.NUM_EPISODES_TO_PLAY = numEpisodesToPlay;
		this.ENEMY_PLAYER_ID = null; // initially

		this.paramFilePath = paramFilePath;

		this.myUnits = null;
		this.enemyUnits = null;
		this.totalRewards = new ArrayList<Double>((int)this.NUM_EPISODES_TO_PLAY / QAgent.NUM_TRAINING_EPISODES_IN_BATCH);
		this.totalRewards.add(0.0);

		this.streamer = Streamer.makeDefaultStreamer(streamerArgString, this.getPlayerNumber());
		this.random = new Random(QAgent.RANDOM_SEED);

		this.qFunctionNN = this.initializeQFunction(loadParams);
		this.lossFunction = new MeanSquaredError();
		this.optimizer = new SGDOptimizer(this.getQFunction().getParameters(),
				QAgent.LEARNING_RATE, 10000); // clip in range [-10000, 10000]
		this.oldInfoPerUnit = new HashMap<Integer, Triple<Matrix, Matrix, Double> >();
		
	}

	private final String getParamFilePath() { return this.paramFilePath; }
	private Integer getEnemyPlayerId() { return this.ENEMY_PLAYER_ID; }
	private Set<Integer> getMyUnitIds() { return this.myUnits; }
	private Set<Integer> getEnemyUnitIds() { return this.enemyUnits; }
	private List<Double> getTotalRewards() { return this.totalRewards; }
	private final Streamer getStreamer() { return this.streamer; }
	private final Random getRandom() { return this.random; }
	
	private Integer teamSize = null;
	private Double reward = null;
	private Map<Integer, Action> initialActions = null;
	private Double maxReward = null;

	/** NN specific stuff **/
	private Model getQFunction() { return this.qFunctionNN; }
	private LossFunction getLossFunction() { return this.lossFunction; }
	private Optimizer getOptimizer() { return this.optimizer; }
	private Map<Integer, Triple<Matrix, Matrix, Double> > getOldInfoPerUnit() { return this.oldInfoPerUnit; }

	private boolean isTrainingEpisode() { return this.numTestEpisodesPlayedInBatch == -1; }

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
	    int feature_dim = 12;
	    
	    int hidden_dim1 = 20; 
	    m.add(new Dense(feature_dim, hidden_dim1, this.getRandom()));
	    // m.add(new Tanh());

	    // layer 2
	    int hidden_dim2 = 20;
	    m.add(new Dense(hidden_dim1, hidden_dim2, this.getRandom()));
	    m.add(new ReLU());
	    
	    // layer 3
	    int hidden_dim3 = 20;
	    m.add(new Dense(hidden_dim2, hidden_dim3, this.getRandom()));
	    m.add(new ReLU());


	    // last layer (must be a scalar)
	    int hidden_dimN = 20; 
	    m.add(new Dense(hidden_dimN, 1, this.getRandom()));
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
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? 
     *
     * Remember that you will need to discount this reward based on the timestep it is received on.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *         damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *         "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param state The current state of the game.
     * @param history History of the episode up until this turn.
     * @param unitId The id of the unit you are looking to calculate the reward for.
     * @return The current reward for that unit
     */
	
	
	
	private double getRewardForUnit(StateView state, HistoryView history, int unitId) {
		
    	Set<Integer> myUnitIdsSet = getMyUnitIds();
    	Set<Integer> enemyUnitIdsSet = getEnemyUnitIds();
    	Integer[] myUnitIds = myUnitIdsSet.toArray(new Integer[myUnitIdsSet.size()]);
    	Integer[] enemyUnitIds = enemyUnitIdsSet.toArray(new Integer[enemyUnitIdsSet.size()]);
    	
    	Integer notNullId = null;
    	
    	for (int enemyId: enemyUnitIds) {
    		if (notNullId != null) {
    			break;
    		}
    		if(state.getUnit(enemyId)!=null) {
    			notNullId = enemyId;
    		}
    	}
    	
    	for (int myUnitId: myUnitIds) {
    		if (notNullId != null) {
    			break;
    		}
    		if(state.getUnit(myUnitId)!=null) {
    			notNullId = myUnitId;
    		}
    	}
		
    	double baseHp = state.getUnit(notNullId).getTemplateView().getBaseHealth();
    	
		double reward = 0.0;
		
		double pDeathPerUnit = -10000;
		double rKillPerUnit = 10000;
		double pPerUnitDamageTaken = -1000;
		double rPerUnitDamageGiven = 1000;
		double rDamageDealtButAlsoRecieved = 500;
		double rDamageDealdAndEnemyFailed = 3000;
		double rDamageDealdAndEnemyIncomplete = 1500;
		double rMoreCompletedActions = 1500;
		double pLessCompletedActions = -1500;

		
		
		if(state.getUnit(unitId) == null)
		{
			reward += pDeathPerUnit * baseHp;
		} else
		{	
			int lastTurnNumber = state.getTurnNumber() - 1;

			// getOldInfoPerUnit - return (featureVector, qValue, reward)
			//getFirst() - featureVector 
			//getSecond() - qValue 
			//getThird() - reward
		
			
			
			
			// reward for good past feature
//			if(state.getTurnNumber() > 0) {
//				//System.out.println(getOldInfoPerUnit().get(unitId).getSecond());
//			}

			
		    
		    if(state.getTurnNumber() > 0)
	    	{
		    	
		    	Map<Integer, ActionResult> enemyActions =  history.getCommandFeedback(getEnemyPlayerId(), lastTurnNumber);
		    	Map<Integer, ActionResult> myActions =  history.getCommandFeedback(getPlayerNumber(), lastTurnNumber);
		    	int numberOfEnemiesCompletedActions = 0;
		    	int numberOfMyCompletedActions = 0;
		    	
		    	Integer[] MyUnitIds = getMyUnitIds().toArray(new Integer[getMyUnitIds().size()]);
		    	Integer[] EnemyUnitIds = getEnemyUnitIds().toArray(new Integer[getEnemyUnitIds().size()]);
		    	
		    	for (Integer enemyId : EnemyUnitIds) {
		    		if(enemyActions.get(enemyId).getFeedback().equals(ActionFeedback.COMPLETED)) {
		    			numberOfEnemiesCompletedActions++;
		    		}
		    	}
		    	
		    	for (Integer myUnitId : MyUnitIds) {
		    		if(myActions.get(myUnitId).getFeedback().equals(ActionFeedback.COMPLETED)) {
		    			numberOfMyCompletedActions++;
		    		}
		    	}
		    
		    	if(numberOfEnemiesCompletedActions > numberOfMyCompletedActions) {
		    		reward += pLessCompletedActions;
		    	} else if(numberOfEnemiesCompletedActions < numberOfMyCompletedActions) {
		    		reward += rMoreCompletedActions;
		    	}
		    	
		    	
	    		for(DeathLog deathLog : history.getDeathLogs(lastTurnNumber))
	    		{
	    			if(deathLog.getController() == getEnemyPlayerId())
	    			{
	    				if(history.getCommandFeedback(getPlayerNumber(), lastTurnNumber).get(unitId).getFeedback().equals(ActionFeedback.COMPLETED)) {
	    					reward += rKillPerUnit * baseHp;
	    				}
	    			}
	    		}
	    		
	    		
	    		for(DamageLog damageLog : history.getDamageLogs(lastTurnNumber))
	    		{
	    		
	    			
	    			if(damageLog.getAttackerID() == unitId)
	    			{
	    				
	    				TargetedAction enemyAction = (TargetedAction) history.getCommandFeedback(getEnemyPlayerId(), lastTurnNumber).get(damageLog.getDefenderID()).getAction();
	    				ActionFeedback enemyActionFB = history.getCommandFeedback(getEnemyPlayerId(), lastTurnNumber).get(damageLog.getDefenderID()).getFeedback();
	    			
	    				if(enemyAction.getTargetId() == unitId && enemyActionFB.equals(ActionFeedback.COMPLETED)) {
	    					reward += damageLog.getDamage() * rDamageDealtButAlsoRecieved;
	    				} else if (enemyAction.getTargetId() == unitId && enemyActionFB.equals(ActionFeedback.FAILED)){
	    					reward += damageLog.getDamage() * rDamageDealdAndEnemyFailed;
	    				} else if(enemyAction.getTargetId() == unitId && enemyActionFB.equals(ActionFeedback.INCOMPLETE)) {
	    					reward += damageLog.getDamage() * rDamageDealdAndEnemyIncomplete;
	    				} else {
	    					reward += damageLog.getDamage() * rPerUnitDamageGiven;
	    				}
	    				
	    				//reward += damageLog.getDamage() * rPerUnitDamageGiven;
	    				
	    			}	
	    			
	    			
	    			if(damageLog.getDefenderID() == unitId)
	    			{
	    				reward += damageLog.getDamage() * pPerUnitDamageTaken;
	    			}
	    		}
	    		
	    	}
		}
	    return reward;
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
    	
        final int NUM_STATIC_FEATURES = 12; 

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
    	
    	for (int i:myUnitIds) {
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
    	
    	
    	
    	for (int i : myUnitIds) {
    		if (i != myUnitId && state.getTurnNumber() > 0 && state.getUnit(i) != null) {
    	    	
    			TargetedAction action = (TargetedAction) actions.get(i).getAction();
    	    	Integer targetId = action.getTargetId();
    	    	
    	    	if(targetId == enemyUnitId) {
    	    		numberOfAtackingUnits++;
    	    	}
    		}
    	}
    	
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
    	double r = this.getRewardForUnit(state, history, atkUnitId);

    	// epsilon-greedy (i.e. random exploration function)
    	if(this.getRandom().nextDouble() < QAgent.EPSILON && this.isTrainingEpisode())
    	{
    		// ignore policy and choose a random action (i.e. attacking which enemy)
    		int randomEnemyIdx = this.getRandom().nextInt(this.getEnemyUnitIds().size());

    		// get the unitId at that position
    		tgtUnitId = this.getEnemyUnitIds().toArray(new Integer[this.getEnemyUnitIds().size()])[randomEnemyIdx];
    		featureVec = this.calculateFeatureVector(state, history, atkUnitId, tgtUnitId);

    		maxQ = this.calculateQValue(featureVec);
    	} else
    	{
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
    	}

    	// remember the info for this unit
    	this.getOldInfoPerUnit().put(atkUnitId, new Triple<Matrix, Matrix, Double>(featureVec, Matrix.full(1, 1, maxQ), r));

    	return tgtUnitId;
    }

    /**
     * This method calculates what the "true" Q(s,a) value should have been based on the Bellman equation for Q-values
     *
     * @param state The current state of the game
     * @param history The current history of the game
     * @param unitId The friendly unitId under consideration
     * @return
     */
    private Matrix getTDGroundTruth(StateView state, HistoryView history, int unitId) throws Exception
    {
    	if(!this.getOldInfoPerUnit().containsKey(unitId))
    	{
    		throw new Exception("unitId=" + unitId + " does not have an old feature vector...cannot calculate TD ground truth for it");
    	}
    	Triple<Matrix, Matrix, Double> oldInfo = this.getOldInfoPerUnit().get(unitId);
    	Double Rs = oldInfo.getThird();

    	double maxQ = Double.NEGATIVE_INFINITY;

    	if(state.getUnit(unitId) != null)
    	{
	    	// try all the actions (i.e. who to attack) in the current state
	    	for(Integer tgtUnitId: this.getEnemyUnitIds())
	    	{
	    		maxQ = Math.max(maxQ, this.calculateQValue(this.calculateFeatureVector(state, history, unitId, tgtUnitId)));
	    	}
    	}
    	else
    	{
    		maxQ = 0.0;
    	}

    	return Matrix.full(1, 1, Rs + QAgent.GAMMA*maxQ); // output is always a scalar in active learning
    }

    /**
     * Calculate the updated weights for this agent. You should construct a matrix
     * @param r The reward R(s) for the prior state
     * @param state Current state of the game.
     * @param history History of the game up until this point
     * @param unitId The unit under consideration
     */
    private void updateParams(StateView state, HistoryView history, int unitId) throws Exception
    {
    	if(!this.getOldInfoPerUnit().containsKey(unitId))
    	{
    		throw new Exception("unitId=" + unitId + " does not have an old feature vector...cannot update params for it");
    	}
    	Triple<Matrix, Matrix, Double> oldInfo = this.getOldInfoPerUnit().get(unitId);
    	Matrix oldFeatureVector = oldInfo.getFirst();
    	Matrix Qsa = oldInfo.getSecond();

    	// reset the optimizer (i.e. reset gradients)
    	this.getOptimizer().reset();

    	// populate gradients
    	this.getQFunction().backwards(oldFeatureVector, this.getLossFunction().backwards(Qsa, this.getTDGroundTruth(state, history, unitId)));

    	// take a step in the correct direction
    	this.getOptimizer().step();
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

    		// calculate the reward for this unit
    		double reward = this.getRewardForUnit(state, history, unitId);

    		// if we are playing a test episode then add these rewards to the total reward for the test games
    		if(this.numTestEpisodesPlayedInBatch != -1)
    		{
    			this.totalRewards.set(this.totalRewards.size() - 1, 
    				this.totalRewards.get(this.totalRewards.size() - 1) + Math.pow(this.GAMMA, state.getTurnNumber() - 1) * reward);
    		}
    		
    		//if this unit does not have an action or the action was completed or failed...give a unit an action
    		if(state.getTurnNumber() == 0 || !prevUnitActions.containsKey(unitId) || 
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.COMPLETED) ||
    				prevUnitActions.get(unitId).getFeedback().equals(ActionFeedback.FAILED))
    		{
    			if(state.getTurnNumber() > 0)
    			{
    				// we have arrived at a new state for that unit, so time to update some gradients
    				try
    				{
						this.updateParams(state, history, unitId);
					} catch (Exception e)
    				{
						System.err.println("QAgent.middleStep [ERROR]: problem updating gradients for transition on unitId=" + unitId);
						e.printStackTrace();
						System.exit(-1);
					}
    			}

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
    	
//    	if(actions.size() > 0)
//    	{
//    		this.getStreamer().streamMove(actions);
//    	}
    	
        return actions;
	}

	@Override
	public void terminalStep(StateView state, HistoryView history)
	{
		if(this.isTrainingEpisode())
		{
			// save the model
			// this.getQFunction().save(this.getParamFilePath());

			this.numTrainingEpisodesPlayed += 1;
			if((this.numTrainingEpisodesPlayed % QAgent.NUM_TRAINING_EPISODES_IN_BATCH) == 0)
			{
				this.numTestEpisodesPlayedInBatch = 0;
			}
		} else
		{
			this.numTestEpisodesPlayedInBatch += 1;
			if((this.numTestEpisodesPlayedInBatch % QAgent.NUM_TESTING_EPISODES_IN_BATCH) == 0)
			{
				this.numTestEpisodesPlayedInBatch = -1;
				// calculate the average
				this.getTotalRewards().set(this.getTotalRewards().size()-1,
						this.getTotalRewards().get(this.getTotalRewards().size()-1) / QAgent.NUM_TRAINING_EPISODES_IN_BATCH);
	
				// print the average test rewards
				this.printTestData(this.getTotalRewards());

				double cumRewardForThisChunkOfTesting = this.getTotalRewards().get(this.getTotalRewards().size() - 1);

				// if cumRewardForThisChunkOfTesting is the max of the list....save the model!
				if(cumRewardForThisChunkOfTesting == this.getMaxTotalReward(this.getTotalRewards()))
				{
					System.out.println(cumRewardForThisChunkOfTesting);
					this.getQFunction().save(this.getParamFilePath());
				}
	
				if(this.numTrainingEpisodesPlayed == this.NUM_EPISODES_TO_PLAY)
				{
					System.out.println("played all " + this.NUM_EPISODES_TO_PLAY + " games!");
					System.exit(0);
				} else
				{
					this.getTotalRewards().add(0.0);
				}
			}
		}
	}
	/**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning curve data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    private void printTestData (List<Double> averageRewards)
    {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++)
        {
            String gamesPlayed = Integer.toString(QAgent.NUM_TRAINING_EPISODES_IN_BATCH*(i+1));
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++)
            {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

	@Override
	public void loadPlayerData(InputStream inStream) {}

	@Override
	public void savePlayerData(OutputStream outStream) {}

}
