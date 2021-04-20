package de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.runnables;

import de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.base.WalkGenerationManager;
import de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.walk_generators.IMidWalkWeightedCapability;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WeightedMidWalkEntityRunnable implements Runnable {


    /**
     * Default Logger
     */
    private static final Logger LOGGER = LoggerFactory.getLogger(WeightedMidWalkEntityRunnable.class);

    /**
     * Entity that is processed by this thread.
     */
    String entity;

    /**
     * Length of each walk.
     */
    int depth;

    /**
     * Number of walks to be performed per entity.
     */
    int numberOfWalks;

    /**
     * The walk generator for which this parser works.
     */
    WalkGenerationManager walkGenerationManager;

    /**
     * Constructor.
     *
     * @param generator     Generator to be used.
     * @param entity        The entity this particular thread shall handle.
     * @param numberOfWalks The number of walks to be performed per entity.
     * @param depth    Desired length of the walk. Defines how many entity steps are allowed. Note that
     *                      this leads to more walk components than the specified depth.
     */
    public WeightedMidWalkEntityRunnable(WalkGenerationManager generator, String entity, int numberOfWalks
            , int depth) {
        this.entity = entity;
        this.numberOfWalks = numberOfWalks;
        this.depth = depth;
        this.walkGenerationManager = generator;
    }

    /**
     * Actual thread execution.
     */
    public void run() {
        if(walkGenerationManager.getWalkGenerator() instanceof IMidWalkWeightedCapability){
            walkGenerationManager.writeToFile(((IMidWalkWeightedCapability) walkGenerationManager.getWalkGenerator()).generateWeightedMidWalksForEntity(walkGenerationManager.shortenUri(entity), this.numberOfWalks, this.depth));
        } else LOGGER.error("NOT YET IMPLEMENTED FOR THE CURRENT WALK GENERATOR!");
    }
}