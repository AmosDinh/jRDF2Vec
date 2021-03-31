package de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.runnables;

import de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.base.IWalkGenerationManager;
import de.uni_mannheim.informatik.dws.jrdf2vec.walk_generation.walk_generators.MemoryWalkGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runnable for walk entity generation.
 */
public class DatatypeWalkEntityProcessingRunnable implements Runnable {


    /**
     * Default Logger
     */
    private static final Logger LOGGER = LoggerFactory.getLogger(DatatypeWalkEntityProcessingRunnable.class);

    /**
     * Entity that is processed by this thread.
     */
    private String entity;

    /**
     * Length of each walk.
     */
    private int depth;

    /**
     * The walk generator for which this parser works.
     */
    private IWalkGenerationManager walkGenerator;

    /**
     * Constructor.
     *
     * @param generator Generator to be used.
     * @param entity    The entity this particular thread shall handle.
     * @param depth     Desired length of the walk. Defines how many entity steps are allowed. Note that
     *                  this leads to more walk components than the specified depth.
     */
    public DatatypeWalkEntityProcessingRunnable(IWalkGenerationManager generator, String entity, int depth) {
        this.entity = entity;
        this.depth = depth;
        this.walkGenerator = generator;
    }

    @Override
    public void run() {
        if (walkGenerator.getWalkGenerator() instanceof MemoryWalkGenerator) {
            // datatype walks are only implemented for memory options
            // yes, the depth and # of walks parameters are this way
            walkGenerator.writeToFile(((MemoryWalkGenerator) walkGenerator.getWalkGenerator()).generateTextWalksForEntity(walkGenerator.shortenUri(entity), this.depth));
        } else LOGGER.error("NOT YET IMPLEMENTED FOR THE CURRENT PARSER!");
    }
}
