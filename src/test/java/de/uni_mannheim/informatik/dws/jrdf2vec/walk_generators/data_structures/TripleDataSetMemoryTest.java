package de.uni_mannheim.informatik.dws.jrdf2vec.walk_generators.data_structures;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class TripleDataSetMemoryTest {

    @Test
     void addDatatypeTriple(){
        TripleDataSetMemory ds = new TripleDataSetMemory();
        ds.addDatatypeTriple("A", "B", "My String");
        assertEquals(0, ds.getObjectTripleSize());
        ds.addDatatypeTriple("A", "C", "My String");
        assertEquals(0, ds.getObjectTripleSize());
        ds.addDatatypeTriple("A", "C", "My String 2");
        assertEquals(0, ds.getObjectTripleSize());

        Map<String, Set<String>> datatypeTuples = ds.getDatatypeTuplesForSubject("A");
        assertEquals(2, datatypeTuples.size());
        assertEquals(1, datatypeTuples.get("B").size());
        assertEquals(2, datatypeTuples.get("C").size());

        Set<String> subjects = ds.getUniqueDatatypeTripleSubjects();
        assertEquals(1, subjects.size());
        assertTrue(subjects.contains("A"));

        subjects = ds.getUniqueSubjects();
        assertEquals(1, subjects.size());
        assertTrue(subjects.contains("A"));
    }

    @Test
    void addObjectPropertyTriple() {
        TripleDataSetMemory ds = new TripleDataSetMemory();
        ds.addObjectTriple("A", "B", "C");
        assertTrue(ds.getObjectTriplesInvolvingObject("C").get(0).subject.equals("A"));
        assertTrue(ds.getObjectTriplesInvolvingPredicate("B").get(0).subject.equals("A"));
        assertTrue(ds.getObjectTriplesInvolvingSubject("A").get(0).object.equals("C"));

        // reference checks
        Triple t1 = new Triple("A", "B", "C");
        Triple t2 = new Triple("A", "B", "C");
        assertFalse(t1 == t2);

        Triple a = ds.getObjectTriplesInvolvingSubject("A").get(0);
        Triple b = ds.getObjectTriplesInvolvingPredicate("B").get(0);
        Triple c = ds.getObjectTriplesInvolvingObject("C").get(0);
        assertTrue((a == b) && (b == c));

        assertEquals(1, ds.getObjectTripleSize());

        Set<String> subjects = ds.getUniqueObjectTripleSubjects();
        assertTrue(subjects.contains("A"));
        assertEquals(1, subjects.size());

        subjects = ds.getUniqueSubjects();
        assertTrue(subjects.contains("A"));
        assertEquals(1, subjects.size());
    }

    @Test
    void addAllObjectPropertyTriple() {
        TripleDataSetMemory ds = new TripleDataSetMemory();
        ds.addObjectTriple("A", "B", "C");
        ds.addObjectTriple("D", "E", "F");

        TripleDataSetMemory ds2 = new TripleDataSetMemory();
        ds2.addObjectTriple("A", "B", "C");
        ds2.addObjectTriple("D", "E", "G");

        ds.addAllObjectTriples(ds2);
        assertEquals(1, ds.getObjectTriplesInvolvingSubject("A").size());
        assertEquals("B", ds.getObjectTriplesInvolvingSubject("A").get(0).predicate);
        assertEquals("C", ds.getObjectTriplesInvolvingSubject("A").get(0).object);

        assertEquals(2, ds.getObjectTriplesInvolvingSubject("D").size());
        assertEquals("E", ds.getObjectTriplesInvolvingSubject("D").get(0).predicate);
        assertTrue(ds.getObjectTriplesInvolvingSubject("D").get(0).object == "F" || ds.getObjectTriplesInvolvingSubject("D").get(0).object == "G");
        assertTrue(ds.getObjectTriplesInvolvingSubject("D").get(1).object == "F" || ds.getObjectTriplesInvolvingSubject("D").get(1).object == "G");
        assertFalse(ds.getObjectTriplesInvolvingSubject("D").get(0).object.equals(ds.getObjectTriplesInvolvingSubject("D").get(1).object));
    }
}