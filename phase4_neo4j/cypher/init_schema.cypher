// =============== CONTRAINTES (uniques) ===============
CREATE CONSTRAINT game_id IF NOT EXISTS FOR (g:Game) REQUIRE g.id IS UNIQUE;
CREATE CONSTRAINT piece_id IF NOT EXISTS FOR (p:Piece) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (r:Rule) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT origin_name IF NOT EXISTS FOR (o:Origin) REQUIRE o.name IS UNIQUE;
CREATE CONSTRAINT mechanic_name IF NOT EXISTS FOR (m:Mechanic) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:GameSession) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT detection_id IF NOT EXISTS FOR (d:Detection) REQUIRE d.id IS UNIQUE;

// =============== INDEX (performance) ===============
CREATE INDEX game_name IF NOT EXISTS FOR (g:Game) ON (g.name);
CREATE INDEX rule_category IF NOT EXISTS FOR (r:Rule) ON (r.category);
CREATE INDEX piece_color IF NOT EXISTS FOR (p:Piece) ON (p.color);