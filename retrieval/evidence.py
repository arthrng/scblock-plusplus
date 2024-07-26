"""
This code is based on the code written by:
Brinkmann, A., Shraga, R., and Bizer, C. (2024). SC-Block: Supervised contrastive blocking within entity resolution pipelines.
In 21st Extended Semantic Web Conference (ESWC 2024), volume 14664 of LCNS, pages 121–142. Springer.

The code can be found at:
https://github.com/wbsg-uni-mannheim/SC-Block
"""

class BaseEvidence:
    """Base class for handling evidence in queries.

    Attributes:
        identifier (int): Unique identifier for the evidence.
        query_table_id (int): ID of the query table.
        entity_id (int): ID of the entity related to the evidence.
        table (str): Table name from which the evidence is derived.
        row_id (str): Row ID in the table.
        context (dict): Context or additional information for the evidence.
        split (str, optional): Indicates the dataset split (train/valid/test). Defaults to None.
        signal (bool, optional): Binary signal indicating relevance of evidence. Defaults to None.
        scale (int, optional): Scale of the evidence. Defaults to None.
        corner_case (bool, optional): Indicates if the evidence is a corner case. Defaults to None.
        similarity_score (float, optional): Similarity score associated with the evidence. Defaults to None.
        seen_training (str, optional): Status of whether the evidence was seen during training. Defaults to None.
        scores (dict): Dictionary to store various scores related to the evidence.
    """

    def __init__(self, identifier, query_table_id, entity_id, table, row_id, context, split=None):
        self.identifier = identifier
        self.query_table_id = query_table_id
        self.entity_id = entity_id
        self.table = table
        self.row_id = row_id
        self.context = context
        self.split = split
        self.signal = None
        self.scale = None
        self.corner_case = None
        self.similarity_score = None
        self.seen_training = None
        self.scores = {}

    def verify(self, signal):
        """Set the signal attribute, ensuring it's defined.

        Args:
            signal (bool): The relevance signal for the evidence.

        Raises:
            ValueError: If signal is None.
        """
        if signal is not None:
            self.signal = signal
        else:
            raise ValueError('The value of signal must be defined (True/False)!')

    def set_context(self, context):
        """Set or update the context attribute for the evidence.

        Args:
            context (dict): The context or additional information.
        """
        self.context = context

    def to_json(self, with_evidence_context, without_score=True):
        """Convert the evidence instance to a JSON-serializable dictionary.

        Args:
            with_evidence_context (bool): Whether to include context in the output.
            without_score (bool, optional): Whether to exclude score-related fields. Defaults to True.

        Returns:
            dict: JSON-serializable dictionary representing the evidence.
        """
        encoded_evidence = {}

        # Convert attributes to camel case and build the dictionary
        for key in self.__dict__.keys():
            camel_cased_key = ''.join([key_part.capitalize() for key_part in key.split('_')])
            camel_cased_key = camel_cased_key[0].lower() + camel_cased_key[1:]
            if camel_cased_key == 'identifier':
                encoded_evidence['id'] = self.__dict__['identifier']
            elif camel_cased_key == 'context':
                if with_evidence_context:
                    encoded_evidence[camel_cased_key] = self.__dict__[key]
            elif without_score and camel_cased_key in ['scores', 'similarityScore']:
                continue
            else:
                encoded_evidence[camel_cased_key] = self.__dict__[key]

        return encoded_evidence

    def __hash__(self):
        """Generate a hash value for the evidence based on entity and row IDs.

        Returns:
            int: Hash value representing the evidence.
        """
        return hash('-'.join([str(self.entity_id), str(self.row_id)]))

    def __str__(self):
        """Provide a string representation of the evidence.

        Returns:
            str: String representation including key attributes.
        """
        return 'Query Table: {}, Entity Id: {}, Table: {}, Row: {}, Signal: {}' \
            .format(self.query_table_id, self.entity_id, self.table, self.row_id, self.signal)

    def __eq__(self, other):
        """Compare two evidence instances for equality.

        Args:
            other (BaseEvidence): Another instance to compare.

        Returns:
            bool: True if equal, otherwise False.
        """
        try:
            return self.__hash__() == other.__hash__()
        except AttributeError:
            return NotImplemented

    def __copy__(self):
        """Create a shallow copy of the evidence instance.

        Returns:
            BaseEvidence: A new instance with the same attributes.
        """
        evidence_copy = BaseEvidence(self.identifier, self.query_table_id, self.entity_id, self.table, self.row_id,
                                     self.context)
        evidence_copy.scale = self.scale
        evidence_copy.signal = self.signal
        evidence_copy.corner_case = self.corner_case
        evidence_copy.similarity_score = self.similarity_score
        evidence_copy.scores = self.scores.copy()

        return evidence_copy

    def aggregate_scores_to_similarity_score(self):
        """Calculate an aggregate similarity score based on individual scores.

        The current implementation averages the scores.
        """
        score_values = [value for value in self.scores.values()]
        if len(score_values) > 0:
            self.similarity_score = sum(score_values) / len(score_values)
        else:
            self.similarity_score = 0

class RetrievalEvidence(BaseEvidence):
    """Specialized class for retrieval-related evidence.

    Inherits from BaseEvidence and adds specific string representation.
    """

    def __init__(self, identifier, query_table_id, entity_id, table, row_id, context, split=None):
        super().__init__(identifier, query_table_id, entity_id, table, row_id, context, split)

    def __str__(self):
        """Provide a string representation including similarity score.

        Returns:
            str: String representation including key attributes and similarity score.
        """
        return 'Query Table: {}, Entity Id: {}, Table: {}, Row: {}, Signal: {}, Similarity: {}' \
            .format(self.query_table_id, self.entity_id, self.table, self.row_id, self.signal, self.similarity_score)

    def __repr__(self):
        """Provide a string representation for debugging.

        Returns:
            str: String representation of the RetrievalEvidence instance.
        """
        return self.__str__()