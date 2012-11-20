DATAPATH='/Volumes/Time Machine Backups/Data/dmytro/'
INSTALLDIR=../../MultiCellPatterns/data

DBFILE=../src/cell_db.h5
OBJECTS=exported_cells.csv events.txt

BUILDDIR=build
INSTALLTARGETS=$(addprefix $(INSTALLDIR)/,$(OBJECTS))
BUILDTARGETS=$(addprefix $(BUILDDIR)/,$(OBJECTS))

all : $(BUILDTARGETS)

install : $(INSTALLTARGETS)

$(BUILDTARGETS) : | $(BUILDDIR)

$(BUILDDIR) : 
	mkdir $(BUILDDIR)

$(INSTALLDIR)/% : $(BUILDDIR)/%
	cp $^ $(INSTALLDIR)

$(BUILDDIR)/events.txt : $(DBFILE) 
	python read_events_from_db.py $< $@

$(BUILDDIR)/exported_cells.csv : $(DBFILE) 
	DATAPATH=$(DATAPATH) python convert_db.py $(DBFILE) $@ 
